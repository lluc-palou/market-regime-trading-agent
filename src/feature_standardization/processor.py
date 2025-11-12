"""
EWMA Half-Life Processor

Main processing logic for EWMA half-life selection.
Implements two-pass processing without keeping all data in memory.
"""

import time
import random
import numpy as np
from datetime import timedelta
from typing import Dict, List
from pyspark.sql import SparkSession

from src.utils.logging import logger

from .ewma_scaler import (
    fit_ewma_scaler,
    apply_ewma_standardization,
    compute_pearson_p_df,
    HALFLIFE_CANDIDATES
)
from .data_loader import (
    get_all_hours,
    load_hour_batch
)


class EWMAHalfLifeProcessor:
    """
    Processes a split to select optimal EWMA half-lives for each feature.
    
    Implements two-pass processing:
    - Pass 1: Stream through training hours, accumulate data, fit EWMA scalers
    - Pass 2: Stream through validation hours, apply scalers, compute P/DF
    
    NO full data accumulation - only processes current hour batch in memory.
    """
    
    def __init__(
        self, 
        spark: SparkSession, 
        db_name: str,
        input_collection_prefix: str = "split_",
        input_collection_suffix: str = "_input",  # Read from transformation output
        train_sample_rate: float = 0.1,
        clip_std: float = 3.0
    ):
        """
        Initialize processor.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            input_collection_prefix: Prefix for split collections (e.g., 'split_')
            input_collection_suffix: Suffix for split collections (e.g., '_output')
            train_sample_rate: Sampling rate for training data (0.1 = 10%)
            clip_std: Standard deviations to clip outliers
        """
        self.spark = spark
        self.db_name = db_name
        self.input_collection_prefix = input_collection_prefix
        self.input_collection_suffix = input_collection_suffix
        self.train_sample_rate = train_sample_rate
        self.clip_std = clip_std
    
    def process_split(self, split_id: int, feature_names: List[str], standardizable_features: List[str]) -> Dict:
        """
        Process a single split with two-pass approach.
        
        Args:
            split_id: Split ID to process
            feature_names: List of ALL feature names (to match features array indices)
            standardizable_features: List of features to actually standardize (subset of feature_names)
            
        Returns:
            Dictionary with results per feature
        """
        logger(f'=' * 80, "INFO")
        logger(f'PROCESSING SPLIT {split_id}', "INFO")
        logger(f'=' * 80, "INFO")
        
        split_collection = f"{self.input_collection_prefix}{split_id}{self.input_collection_suffix}"
        logger(f'Reading from collection: {split_collection}', "INFO")
        logger(f'Total features: {len(feature_names)}', "INFO")
        logger(f'Standardizable features: {len(standardizable_features)}', "INFO")
        logger(f'Training sample rate: {self.train_sample_rate * 100:.1f}%', "INFO")
        logger(f'Half-life candidates: {HALFLIFE_CANDIDATES}', "INFO")
        
        # Discover hours
        all_hours = get_all_hours(self.spark, self.db_name, split_collection)
        
        if not all_hours:
            logger(f'No hours found for split {split_id}', "ERROR")
            return {}
        
        # PASS 1: Fit EWMA scalers on training data
        logger(f'Pass 1: Fitting EWMA scalers on training data...', "INFO")
        fitted_scalers = self._pass1_fit_scalers(
            split_collection, all_hours, feature_names, standardizable_features
        )
        
        # PASS 2: Evaluate scalers on validation data
        logger(f'Pass 2: Evaluating scalers on validation data...', "INFO")
        results = self._pass2_evaluate_scalers(
            split_collection, all_hours, feature_names, standardizable_features, fitted_scalers
        )
        
        return results
    
    def _pass1_fit_scalers(
        self, 
        split_collection: str, 
        all_hours: List,
        feature_names: List[str],
        standardizable_features: List[str]
    ) -> Dict:
        """
        Pass 1: Fit EWMA scalers by accumulating training data per hour.
        
        Strategy: For each hour, accumulate training samples, then aggregate
        across hours to fit final EWMA scalers. Memory efficient.
        
        Args:
            split_collection: Split collection name
            all_hours: List of hour datetimes
            feature_names: List of ALL feature names (for array indexing)
            standardizable_features: List of features to standardize (subset)
            
        Returns:
            Dictionary of fitted EWMA scalers per feature per half-life
        """
        # Initialize accumulators only for standardizable features
        train_accumulators = {feat: [] for feat in standardizable_features}
        train_sample_counts = {feat: 0 for feat in standardizable_features}
        
        # Track nulls for summary
        null_counts = {feat: 0 for feat in standardizable_features}
        total_samples_seen = 0
        
        hours_processed = 0
        
        for i, hour in enumerate(all_hours):
            hour_start = time.time()
            end_hour = hour + timedelta(hours=1)
            
            # Load hour batch
            df = load_hour_batch(
                self.spark,
                self.db_name,
                split_collection,
                hour,
                end_hour
            )
            
            batch_count = df.count()
            
            if batch_count == 0:
                continue
            
            # Collect to driver
            rows = df.select("features", "role").collect()
            
            # DEBUG: Check first row structure
            if i == 0 and len(rows) > 0:
                first_row = rows[0]
                logger(f'DEBUG - First row keys: {first_row.asDict().keys()}', "INFO")
                logger(f'DEBUG - Features type: {type(first_row["features"])}', "INFO")
                logger(f'DEBUG - Features length: {len(first_row["features"]) if first_row["features"] else "None"}', "INFO")
                logger(f'DEBUG - Role: {first_row["role"]}', "INFO")
                if first_row["features"]:
                    logger(f'DEBUG - First 3 features: {first_row["features"][:3]}', "INFO")
            
            # Process hour batch - accumulate training samples
            hour_samples = {feat: [] for feat in standardizable_features}
            
            for row in rows:
                features_array = row['features']
                role = row['role']
                
                # Only process training data in pass 1 (train or train_warmup)
                if role not in ('train', 'train_warmup'):
                    continue
                
                total_samples_seen += 1
                
                # Apply sampling
                if random.random() > self.train_sample_rate:
                    continue
                
                # Validate features array
                if features_array is None:
                    logger(f'DEBUG - Features array is None for role={role}', "WARNING")
                    continue
                
                if len(features_array) != len(feature_names):
                    logger(f'DEBUG - Features array length mismatch: {len(features_array)} vs {len(feature_names)}', "WARNING")
                    continue
                
                # Process only standardizable features
                for feat_idx, feat_name in enumerate(feature_names):
                    # Skip if not in standardizable list
                    if feat_name not in standardizable_features:
                        continue
                    
                    raw_value = features_array[feat_idx]
                    
                    if raw_value is None:
                        null_counts[feat_name] += 1
                        continue
                    
                    try:
                        value = float(raw_value)
                        if np.isfinite(value):
                            hour_samples[feat_name].append(value)
                    except (TypeError, ValueError):
                        null_counts[feat_name] += 1
                        continue
            
            # Add hour samples to overall accumulators
            for feat_name in standardizable_features:
                if hour_samples[feat_name]:
                    train_accumulators[feat_name].extend(hour_samples[feat_name])
                    train_sample_counts[feat_name] += len(hour_samples[feat_name])
            
            hours_processed += 1
            
            # Log progress
            hour_duration = time.time() - hour_start
            if (i + 1) % 10 == 0 or (i + 1) == len(all_hours):
                logger(f'Pass 1 - Processed hour {i+1}/{len(all_hours)} '
                       f'({hour.strftime("%Y-%m-%d %H:%M")}) in {hour_duration:.2f}s',
                       "INFO")
        
        # Summary statistics
        logger(f'Pass 1 - Processed {hours_processed} hours', "INFO")
        logger(f'Pass 1 - Total training samples seen: {total_samples_seen:,}', "INFO")
        
        # Show sample counts per feature
        sample_counts_list = [(feat, train_sample_counts[feat]) 
                              for feat in list(standardizable_features)[:5]]
        for feat, count in sample_counts_list:
            logger(f'  {feat}: {count:,} samples', "INFO")
        if len(standardizable_features) > 5:
            logger(f'  ... and {len(standardizable_features) - 5} more features', "INFO")
        
        # Null summary
        features_with_nulls = {feat: count for feat, count in null_counts.items() if count > 0}
        if features_with_nulls:
            logger(f'Pass 1 - Null summary: {len(features_with_nulls)} features had nulls', "WARNING")
            sorted_nulls = sorted(features_with_nulls.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, count in sorted_nulls:
                pct = 100 * count / total_samples_seen if total_samples_seen > 0 else 0
                logger(f'  {feat}: {count:,} nulls ({pct:.2f}%)', "WARNING")
        
        # Fit EWMA scalers for each half-life
        logger(f'Pass 1 - Fitting EWMA scalers for each half-life...', "INFO")
        
        fitted_scalers = {}
        
        for feat_name in standardizable_features:
            train_data = np.array(train_accumulators[feat_name])
            
            if len(train_data) < 20:
                logger(f'Pass 1 - Skipping {feat_name}: insufficient training data '
                      f'({len(train_data)} samples)', "WARNING")
                continue
            
            # Fit scaler for each half-life candidate
            fitted_scalers[feat_name] = {}
            
            for half_life in HALFLIFE_CANDIDATES:
                scaler = fit_ewma_scaler(train_data, half_life)
                if scaler is not None:
                    fitted_scalers[feat_name][half_life] = scaler
            
            logger(f'Pass 1 - {feat_name}: fitted {len(fitted_scalers[feat_name])} scalers '
                  f'({len(train_data):,} samples)', "INFO")
        
        return fitted_scalers
    
    def _pass2_evaluate_scalers(
        self,
        split_collection: str,
        all_hours: List,
        feature_names: List[str],
        standardizable_features: List[str],
        fitted_scalers: Dict
    ) -> Dict:
        """
        Pass 2: Evaluate fitted EWMA scalers on validation data.
        
        Strategy: Stream through validation hours, accumulate standardized values
        per hour, compute final P/DF scores. Memory efficient.
        
        Args:
            split_collection: Split collection name
            all_hours: List of hour datetimes
            feature_names: List of ALL feature names (for array indexing)
            standardizable_features: List of features to standardize (subset)
            fitted_scalers: Fitted EWMA scalers from pass 1
            
        Returns:
            Dictionary with evaluation results per feature
        """
        # Initialize validation accumulators (accumulate standardized values)
        validation_accumulators = {
            feat: {half_life: [] for half_life in HALFLIFE_CANDIDATES}
            for feat in standardizable_features
        }
        
        # Track nulls for summary
        val_null_counts = {feat: 0 for feat in standardizable_features}
        val_samples_seen = 0
        
        hours_processed = 0
        
        for i, hour in enumerate(all_hours):
            hour_start = time.time()
            end_hour = hour + timedelta(hours=1)
            
            # Load hour batch
            df = load_hour_batch(
                self.spark,
                self.db_name,
                split_collection,
                hour,
                end_hour
            )
            
            batch_count = df.count()
            
            if batch_count == 0:
                continue
            
            # Collect to driver
            rows = df.select("features", "role").collect()
            
            # Process hour batch - apply scalers to validation samples
            for row in rows:
                features_array = row['features']
                role = row['role']
                
                # Only process validation data in pass 2
                if role != 'validation':
                    continue
                
                val_samples_seen += 1
                
                # Validate features array
                if features_array is None or len(features_array) != len(feature_names):
                    continue
                
                # Process only standardizable features
                for feat_idx, feat_name in enumerate(feature_names):
                    # Skip if not in standardizable list or no fitted scaler
                    if feat_name not in standardizable_features or feat_name not in fitted_scalers:
                        continue
                    
                    raw_value = features_array[feat_idx]
                    
                    if raw_value is None:
                        val_null_counts[feat_name] += 1
                        continue
                    
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        val_null_counts[feat_name] += 1
                        continue
                    
                    if not np.isfinite(value):
                        continue
                    
                    # Apply each fitted scaler
                    for half_life, scaler in fitted_scalers[feat_name].items():
                        standardized = scaler.standardize(value, clip_std=self.clip_std)
                        
                        if np.isfinite(standardized):
                            validation_accumulators[feat_name][half_life].append(standardized)
            
            hours_processed += 1
            
            # Log progress
            hour_duration = time.time() - hour_start
            if (i + 1) % 10 == 0 or (i + 1) == len(all_hours):
                logger(f'Pass 2 - Processed hour {i+1}/{len(all_hours)} '
                       f'({hour.strftime("%Y-%m-%d %H:%M")}) in {hour_duration:.2f}s',
                       "INFO")
        
        # Compute P/DF scores for each half-life
        logger(f'Pass 2 - Computing P/DF scores...', "INFO")
        
        # Log null summary
        features_with_nulls = {feat: count for feat, count in val_null_counts.items() if count > 0}
        if features_with_nulls:
            logger(f'Pass 2 - Null summary: {len(features_with_nulls)} features had nulls', "WARNING")
            sorted_nulls = sorted(features_with_nulls.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, count in sorted_nulls:
                pct = 100 * count / val_samples_seen if val_samples_seen > 0 else 0
                logger(f'  {feat}: {count:,} nulls ({pct:.2f}% of {val_samples_seen:,} validation samples)',
                       "WARNING")
        else:
            logger(f'Pass 2 - No nulls found in validation data [OK]', "INFO")
        
        results = {}
        
        for feat_name in standardizable_features:
            if feat_name not in fitted_scalers:
                continue
            
            feature_results = {
                'fitted_params': {},
                'validation_scores': {},
                'val_samples': {}
            }
            
            # Store fitted parameters
            for half_life, scaler in fitted_scalers[feat_name].items():
                feature_results['fitted_params'][half_life] = scaler.get_params()
            
            # Compute P/DF for each half-life
            for half_life in HALFLIFE_CANDIDATES:
                if half_life not in fitted_scalers[feat_name]:
                    continue
                
                val_standardized = np.array(validation_accumulators[feat_name][half_life])
                
                if len(val_standardized) < 20:
                    continue
                
                p_df = compute_pearson_p_df(val_standardized)
                feature_results['validation_scores'][half_life] = p_df
                feature_results['val_samples'][half_life] = len(val_standardized)
            
            # Select best half-life
            if feature_results['validation_scores']:
                best_half_life = min(
                    feature_results['validation_scores'],
                    key=feature_results['validation_scores'].get
                )
                feature_results['best_half_life'] = best_half_life
                feature_results['best_score'] = feature_results['validation_scores'][best_half_life]
                
                logger(f'Pass 2 - {feat_name}: best_half_life={best_half_life} '
                      f'(P/DF={feature_results["best_score"]:.3f})', "INFO")
            else:
                logger(f'Pass 2 - {feat_name}: no valid half-lives', "WARNING")
            
            results[feat_name] = feature_results
        
        return results