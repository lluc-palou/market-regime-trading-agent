"""
EWMA Standardization Application

Applies EWMA standardization hour-by-hour using selected half-life parameters.
Uses sequential EWMA state maintenance for proper temporal processing.
"""

import json
import time
from datetime import timedelta
from typing import Dict, List
import numpy as np
from pyspark.sql import SparkSession

from src.utils.logging import logger
from src.feature_standardization.data_loader import get_all_hours, load_hour_batch
from src.feature_standardization.ewma_scaler import EWMAScaler


class EWMAStandardizationApplicator:
    """
    Applies EWMA standardization to split data using selected half-lives.
    
    Maintains EWMA state across hours to ensure proper sequential processing.
    """
    
    def __init__(
        self,
        spark: SparkSession,
        db_name: str,
        final_halflifes: Dict[str, int],
        clip_std: float = 3.0
    ):
        """
        Initialize applicator.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            final_halflifes: Dictionary mapping feature_name -> half_life
            clip_std: Standard deviations to clip outliers
        """
        self.spark = spark
        self.db_name = db_name
        self.final_halflifes = final_halflifes
        self.clip_std = clip_std
        
        # Initialize EWMA scalers for each feature
        self.scalers = {}
        for feat_name, half_life in final_halflifes.items():
            self.scalers[feat_name] = EWMAScaler(half_life)
    
    def apply_to_split(
        self,
        split_id: int,
        feature_names: List[str],
        input_collection_prefix: str = "split_",
        input_collection_suffix: str = "_input",
        output_collection_prefix: str = "split_",
        output_collection_suffix: str = "_output"
    ):
        """
        Apply EWMA standardization to a single split.
        
        Processes hours sequentially to maintain EWMA state.
        Writes to output collection in cyclic pattern.
        
        Args:
            split_id: Split ID to process
            feature_names: List of ALL feature names (for array indexing)
            input_collection_prefix: Input collection prefix
            input_collection_suffix: Input collection suffix
            output_collection_prefix: Output collection prefix
            output_collection_suffix: Output collection suffix
        """
        logger(f'=' * 80, "INFO")
        logger(f'APPLYING EWMA STANDARDIZATION - SPLIT {split_id}', "INFO")
        logger(f'=' * 80, "INFO")
        
        input_collection = f"{input_collection_prefix}{split_id}{input_collection_suffix}"
        output_collection = f"{output_collection_prefix}{split_id}{output_collection_suffix}"
        
        logger(f'Input: {input_collection}', "INFO")
        logger(f'Output: {output_collection}', "INFO")
        logger(f'Standardizing {len(self.final_halflifes)} features', "INFO")
        
        # Get all hours
        all_hours = get_all_hours(self.spark, self.db_name, input_collection)
        
        if not all_hours:
            logger(f'No hours found for split {split_id}', "ERROR")
            return
        
        logger(f'Processing {len(all_hours)} hours sequentially', "INFO")
        
        # Process hours sequentially to maintain EWMA state
        total_processed = 0
        first_batch = True
        
        for hour_idx, start_hour in enumerate(all_hours):
            hour_start = time.time()
            end_hour = start_hour + timedelta(hours=1)
            
            # Load hour batch
            hour_df = load_hour_batch(
                self.spark,
                self.db_name,
                input_collection,
                start_hour,
                end_hour
            )
            
            # Collect hour data (sorted by timestamp)
            # Skip .count() to avoid extra DataFrame scan - we'll check after collect
            rows = hour_df.collect()

            if not rows:
                continue

            hour_count = len(rows)
            
            # Process rows sequentially
            transformed_rows = []
            
            for row in rows:
                row_dict = row.asDict()
                
                # Handle timestamp timezone
                ts = row_dict.get('timestamp')
                if ts and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    import pytz
                    if ts.tzinfo != pytz.UTC:
                        ts = ts.astimezone(pytz.UTC)
                    ts = ts.replace(tzinfo=None)
                    row_dict['timestamp'] = ts
                
                features = row_dict.get('features')
                role = row_dict.get('role')
                
                if features is None:
                    transformed_rows.append(row_dict)
                    continue
                
                # Apply EWMA standardization - vectorized approach
                features_array = np.array(features, dtype=np.float64)

                # Update EWMA and standardize each feature
                for feat_idx, feat_name in enumerate(feature_names):
                    # Skip if not in standardization list
                    if feat_name not in self.scalers:
                        continue

                    if feat_idx >= len(features_array):
                        continue

                    raw_value = features_array[feat_idx]

                    if not np.isfinite(raw_value):
                        continue

                    scaler = self.scalers[feat_name]

                    # Update EWMA state (sequential learning)
                    # For training data, update the scaler
                    if role in ('train', 'train_warmup'):
                        scaler.update(raw_value)

                    # Standardize using current EWMA state
                    standardized = scaler.standardize(raw_value, clip_std=self.clip_std)

                    if np.isfinite(standardized):
                        features_array[feat_idx] = standardized

                # Convert to Python list (NumPy arrays can't go directly to Spark)
                row_dict['features'] = features_array.tolist()
                transformed_rows.append(row_dict)
            
            if not transformed_rows:
                logger(f'  Hour {hour_idx + 1}/{len(all_hours)}: No data to write', "WARNING")
                continue
            
            # Convert back to DataFrame
            transformed_df = self.spark.createDataFrame(transformed_rows, schema=hour_df.schema)
            
            # Drop _id if present
            if '_id' in transformed_df.columns:
                transformed_df = transformed_df.drop('_id')
            
            # Sort by timestamp
            transformed_df = transformed_df.orderBy("timestamp")
            
            # Write with ordered writes
            write_mode = "overwrite" if first_batch else "append"
            
            (transformed_df.write.format("mongodb")
             .option("database", self.db_name)
             .option("collection", output_collection)
             .option("ordered", "true")
             .mode(write_mode)
             .save())
            
            total_processed += len(transformed_rows)
            first_batch = False
            
            hour_duration = time.time() - hour_start
            
            if (hour_idx + 1) % 10 == 0 or (hour_idx + 1) == len(all_hours):
                logger(f'  Hour {hour_idx + 1}/{len(all_hours)} '
                       f'({start_hour.strftime("%Y-%m-%d %H:%M")}): '
                       f'{len(transformed_rows):,} samples in {hour_duration:.2f}s',
                       "INFO")
        
        logger(f'Split {split_id}: Processed {total_processed:,} documents', "INFO")
        logger(f'Written to {output_collection}', "INFO")
    
    def get_scaler_states(self) -> Dict:
        """
        Get current EWMA scaler states.
        
        Returns:
            Dictionary with scaler states per feature
        """
        states = {}
        for feat_name, scaler in self.scalers.items():
            states[feat_name] = scaler.get_params()
        return states