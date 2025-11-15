"""
Feature Transform Processor

Main processing logic for feature transformation selection.
Implements TRUE two-pass processing without keeping all data in memory.
"""

import time
import random
import numpy as np
from datetime import timedelta
from typing import Dict, List
from pyspark.sql import SparkSession

from src.utils.logging import logger

from .transforms import (
    fit_transform_params,
    apply_transform,
    compute_pearson_p_df,
    TRANSFORM_TYPES
)
from .data_loader import (
    get_all_hours,
    load_hour_batch
)


class FeatureTransformProcessor:
    """
    Processes a split to select optimal transformations for each feature.
    
    Implements TRUE two-pass processing:
    - Pass 1: Stream through training hours, accumulate per hour, fit transforms
    - Pass 2: Stream through validation hours, apply transforms, compute P/DF
    
    NO full data accumulation - only processes current hour batch in memory.
    """
    
    def __init__(self, spark: SparkSession, db_name: str,
                 input_collection_prefix: str, input_collection_suffix: str = "",
                 train_sample_rate: float = 0.1):
        """
        Initialize processor.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            input_collection_prefix: Prefix for split collections (e.g., 'split_')
            input_collection_suffix: Suffix for split collections (e.g., '_input')
            train_sample_rate: Sampling rate for training data
        """
        self.spark = spark
        self.db_name = db_name
        self.input_collection_prefix = input_collection_prefix
        self.input_collection_suffix = input_collection_suffix
        self.train_sample_rate = train_sample_rate
    
    def process_split(self, split_id: int, feature_names: List[str],
                     all_feature_names: List[str] = None) -> Dict:
        """
        Process a single split with true two-pass approach.

        Args:
            split_id: Split ID to process
            feature_names: List of feature names to transform (subset)
            all_feature_names: Full list of feature names from DB (for array validation)
                              If None, assumes feature_names is the full list

        Returns:
            Dictionary with results per feature
        """
        # Use all_feature_names for validation, feature_names for processing
        if all_feature_names is None:
            all_feature_names = feature_names
        logger(f'=' * 80, "INFO")
        logger(f'PROCESSING SPLIT {split_id}', "INFO")
        logger(f'=' * 80, "INFO")
        
        split_collection = f"{self.input_collection_prefix}{split_id}{self.input_collection_suffix}"
        logger(f'Reading from collection: {split_collection}', "INFO")
        
        # Discover hours
        all_hours = get_all_hours(self.spark, self.db_name, split_collection)
        
        if not all_hours:
            logger(f'No hours found for split {split_id}', "ERROR")
            return {}
        
        # PASS 1: Fit transformations on training data
        logger(f'Pass 1: Fitting transformations on training data...', "INFO")
        fitted_transforms = self._pass1_fit_transforms(
            split_collection, all_hours, feature_names, all_feature_names
        )

        # PASS 2: Evaluate transformations on validation data
        logger(f'Pass 2: Evaluating transformations on validation data...', "INFO")
        results = self._pass2_evaluate_transforms(
            split_collection, all_hours, feature_names, fitted_transforms, all_feature_names
        )
        
        return results
    
    def _pass1_fit_transforms(self, split_collection: str, all_hours: List,
                             feature_names: List[str], all_feature_names: List[str]) -> Dict:
        """
        Pass 1: Fit transformations by accumulating training data per hour.

        Strategy: For each hour, accumulate training samples, then aggregate
        across hours to fit final transformations. Memory efficient.

        Args:
            split_collection: Split collection name
            all_hours: List of hour datetimes
            feature_names: List of transformable feature names (subset)
            all_feature_names: Full list of feature names from DB (for validation)

        Returns:
            Dictionary of fitted transformation parameters per feature
        """
        # Build index mapping from feature name to array index
        feature_indices = {name: all_feature_names.index(name) for name in feature_names}
        # Initialize online accumulators (for computing mean/std incrementally)
        train_accumulators = {feat: [] for feat in feature_names}
        train_sample_counts = {feat: 0 for feat in feature_names}
        
        # Track nulls for summary reporting
        null_counts = {feat: 0 for feat in feature_names}
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
            
            # Process hour batch - accumulate training samples
            hour_samples = {feat: [] for feat in feature_names}
            
            for row in rows:
                features_array = row['features']
                role = row['role']
                
                # Only process training data in pass 1 - EXPLICIT filter
                if role != 'train':
                    continue
                
                total_samples_seen += 1
                
                # Apply sampling
                if random.random() > self.train_sample_rate:
                    continue
                
                # Validate features array against full feature list
                if features_array is None or len(features_array) != len(all_feature_names):
                    logger(f'Pass 1 - Invalid features array (expected {len(all_feature_names)}, '
                          f'got {len(features_array) if features_array else "None"}) for role={role}',
                          "WARNING")
                    continue

                # Check if sample has ANY nulls in transformable features - if so, skip entire sample
                has_nulls = False
                for feat_name in feature_names:
                    feat_idx = feature_indices[feat_name]
                    if features_array[feat_idx] is None:
                        has_nulls = True
                        null_counts[feat_name] += 1
                
                if has_nulls:
                    continue  # Skip this entire sample

                # Accumulate samples for this hour (no nulls present)
                for feat_name in feature_names:
                    feat_idx = feature_indices[feat_name]
                    raw_value = features_array[feat_idx]
                    
                    try:
                        value = float(raw_value)
                        if np.isfinite(value):
                            hour_samples[feat_name].append(value)
                    except (TypeError, ValueError) as e:
                        logger(f'Pass 1 - Cannot convert feature {feat_name} value to float: {e}',
                              "WARNING")
                        # Don't add this sample to any feature
                        break
            
            # Add hour samples to overall accumulators
            for feat_name in feature_names:
                if hour_samples[feat_name]:
                    train_accumulators[feat_name].extend(hour_samples[feat_name])
                    train_sample_counts[feat_name] += len(hour_samples[feat_name])
            
            hours_processed += 1
            
            # Log progress
            hour_duration = time.time() - hour_start
            if (i + 1) % 5 == 0 or (i + 1) == len(all_hours):
                logger(f'Pass 1 - Processed hour {i+1}/{len(all_hours)} '
                       f'({hour.strftime("%Y-%m-%d %H:%M")}) in {hour_duration:.2f}s', 
                       "INFO")
        
        # Now fit transformations on accumulated training data
        logger(f'Pass 1 - Fitting transformations...', "INFO")
        
        # Log null summary
        features_with_nulls = {feat: count for feat, count in null_counts.items() if count > 0}
        if features_with_nulls:
            logger(f'Pass 1 - Null summary: {len(features_with_nulls)} features had nulls', "WARNING")
            # Show top 5 features with most nulls
            sorted_nulls = sorted(features_with_nulls.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, count in sorted_nulls:
                pct = 100 * count / total_samples_seen if total_samples_seen > 0 else 0
                logger(f'  {feat}: {count:,} nulls ({pct:.2f}% of {total_samples_seen:,} train samples)', 
                       "WARNING")
            if len(features_with_nulls) > 5:
                logger(f'  ... and {len(features_with_nulls) - 5} more features', "WARNING")
        else:
            logger(f'Pass 1 - No nulls found in training data', "INFO")
        
        fitted_transforms = {}
        
        for feat_name in feature_names:
            train_data = np.array(train_accumulators[feat_name])
            
            if len(train_data) < 20:
                logger(f'Pass 1 - Skipping {feat_name}: insufficient training data '
                      f'({len(train_data)} samples)', "WARNING")
                continue
            
            # Fit each transformation type
            fitted_transforms[feat_name] = {}
            
            for transform_type in TRANSFORM_TYPES:
                params = fit_transform_params(train_data, transform_type)
                if params is not None:
                    fitted_transforms[feat_name][transform_type] = params
            
            logger(f'Pass 1 - {feat_name}: fitted {len(fitted_transforms[feat_name])} transforms '
                  f'({len(train_data)} samples)', "INFO")
        
        return fitted_transforms
    
    def _pass2_evaluate_transforms(self, split_collection: str, all_hours: List,
                                  feature_names: List[str],
                                  fitted_transforms: Dict, all_feature_names: List[str]) -> Dict:
        """
        Pass 2: Evaluate fitted transformations on validation data.

        Strategy: Stream through validation hours, accumulate transformed values
        per hour, compute final P/DF scores. Memory efficient.

        Args:
            split_collection: Split collection name
            all_hours: List of hour datetimes
            feature_names: List of transformable feature names (subset)
            fitted_transforms: Fitted transformation parameters from pass 1
            all_feature_names: Full list of feature names from DB (for validation)

        Returns:
            Dictionary with evaluation results per feature
        """
        # Build index mapping from feature name to array index
        feature_indices = {name: all_feature_names.index(name) for name in feature_names}
        # Initialize validation accumulators (accumulate transformed values)
        validation_accumulators = {
            feat: {transform: [] for transform in TRANSFORM_TYPES}
            for feat in feature_names
        }
        
        # Track nulls for summary
        val_null_counts = {feat: 0 for feat in feature_names}
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
            
            # Process hour batch - apply transformations to validation samples
            for row in rows:
                features_array = row['features']
                role = row['role']
                
                # Only process validation data in pass 2 - EXPLICIT filter
                if role != 'validation':
                    continue
                
                val_samples_seen += 1
                
                # Validate features array against full feature list
                if features_array is None or len(features_array) != len(all_feature_names):
                    logger(f'Pass 2 - Invalid features array (expected {len(all_feature_names)}, '
                          f'got {len(features_array) if features_array else "None"}) for role={role}',
                          "WARNING")
                    continue

                # Check if sample has ANY nulls in transformable features - if so, skip entire sample
                has_nulls = False
                for feat_name in feature_names:
                    feat_idx = feature_indices[feat_name]
                    if features_array[feat_idx] is None:
                        has_nulls = True
                        val_null_counts[feat_name] += 1
                
                if has_nulls:
                    continue  # Skip this entire sample

                # Process each feature (no nulls present)
                for feat_name in feature_names:
                    if feat_name not in fitted_transforms:
                        continue

                    feat_idx = feature_indices[feat_name]
                    raw_value = features_array[feat_idx]
                    
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError) as e:
                        logger(f'Pass 2 - Cannot convert feature {feat_name} value to float: {e}',
                              "WARNING")
                        break  # Skip entire sample
                    
                    if not np.isfinite(value):
                        break  # Skip entire sample if any non-finite value
                    
                    # Apply each fitted transformation
                    for transform_type, params in fitted_transforms[feat_name].items():
                        transformed = apply_transform(np.array([value]), params)[0]
                        
                        if np.isfinite(transformed):
                            validation_accumulators[feat_name][transform_type].append(transformed)
            
            hours_processed += 1
            
            # Log progress
            hour_duration = time.time() - hour_start
            if (i + 1) % 5 == 0 or (i + 1) == len(all_hours):
                logger(f'Pass 2 - Processed hour {i+1}/{len(all_hours)} '
                       f'({hour.strftime("%Y-%m-%d %H:%M")}) in {hour_duration:.2f}s',
                       "INFO")
        
        # Compute P/DF scores for each transformation
        logger(f'Pass 2 - Computing P/DF scores...', "INFO")
        
        # Log null summary
        features_with_nulls = {feat: count for feat, count in val_null_counts.items() if count > 0}
        if features_with_nulls:
            logger(f'Pass 2 - Null summary: {len(features_with_nulls)} features had nulls', "WARNING")
            # Show top 5 features with most nulls
            sorted_nulls = sorted(features_with_nulls.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, count in sorted_nulls:
                pct = 100 * count / val_samples_seen if val_samples_seen > 0 else 0
                logger(f'  {feat}: {count:,} nulls ({pct:.2f}% of {val_samples_seen:,} validation samples)', 
                       "WARNING")
            if len(features_with_nulls) > 5:
                logger(f'  ... and {len(features_with_nulls) - 5} more features', "WARNING")
        else:
            logger(f'Pass 2 - No nulls found in validation data', "INFO")
        
        results = {}
        
        for feat_name in feature_names:
            if feat_name not in fitted_transforms:
                continue
            
            feature_results = {
                'fitted_params': fitted_transforms[feat_name],
                'validation_scores': {},
                'val_samples': {}
            }
            
            # Compute P/DF for each transformation
            for transform_type in TRANSFORM_TYPES:
                if transform_type not in fitted_transforms[feat_name]:
                    continue
                
                val_transformed = np.array(validation_accumulators[feat_name][transform_type])
                
                if len(val_transformed) < 20:
                    continue
                
                p_df = compute_pearson_p_df(val_transformed)
                feature_results['validation_scores'][transform_type] = p_df
                feature_results['val_samples'][transform_type] = len(val_transformed)
            
            # Select best transformation
            if feature_results['validation_scores']:
                best_transform = min(
                    feature_results['validation_scores'],
                    key=feature_results['validation_scores'].get
                )
                feature_results['best_transform'] = best_transform
                feature_results['best_score'] = feature_results['validation_scores'][best_transform]
                
                logger(f'Pass 2 - {feat_name}: best={best_transform} '
                      f'(P/DF={feature_results["best_score"]:.3f})', "INFO")
            else:
                logger(f'Pass 2 - {feat_name}: no valid transformations', "WARNING")
            
            results[feat_name] = feature_results
        
        return results