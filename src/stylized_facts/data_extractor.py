"""
Streaming Window Data Extractor

Loads window data hour-by-hour from split collections for memory efficiency.
Properly extracts features from array format and maps to feature names.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.utils.logging import logger
from src.utils.timestamp import parse_timestamp_from_metadata, format_timestamp_for_mongodb
from src.utils.database import read_all_with_timestamp_strings


def parse_timestamp(ts) -> datetime:
    """
    Parse timestamp to naive datetime (UTC).
    
    Handles multiple input types from metadata YAML.
    
    Args:
        ts: Timestamp in various formats (str, datetime, pandas Timestamp)
        
    Returns:
        Naive datetime object
    """
    return parse_timestamp_from_metadata(ts)


class StreamingWindowExtractor:
    """
    Extracts window data hour-by-hour to manage memory efficiently.
    
    Properly handles array-based feature storage:
    - features: Array of feature values
    - feature_names: Array of feature names
    
    Converts to wide-format DataFrame with feature names as columns.
    """
    
    def __init__(self, spark: SparkSession, db_name: str):
        """
        Initialize streaming extractor.
        
        Args:
            spark: SparkSession instance
            db_name: MongoDB database name
        """
        self.spark = spark
        self.db_name = db_name
        self.feature_names_cache = None  # Cache feature names from first load
    
    def _get_hours_in_window(
        self, 
        window_start: datetime, 
        window_end: datetime
    ) -> List[datetime]:
        """
        Get all hour boundaries that overlap with window.
        
        Args:
            window_start: Window start timestamp
            window_end: Window end timestamp
            
        Returns:
            List of hour boundary timestamps
        """
        # Round down to hour boundary
        current = window_start.replace(minute=0, second=0, microsecond=0)
        end_boundary = window_end.replace(minute=0, second=0, microsecond=0)
        
        hours = []
        while current <= end_boundary:
            hours.append(current)
            current += timedelta(hours=1)
        
        return hours
    
    def _load_hour_batch(
        self,
        collection_name: str,
        start_hour: datetime,
        end_hour: datetime
    ) -> pd.DataFrame:
        """
        Load data for one hour batch from MongoDB.
        
        Extracts features array and feature_names, converts to wide format.
        
        Args:
            collection_name: MongoDB collection name
            start_hour: Hour start boundary
            end_hour: Hour end boundary
            
        Returns:
            Pandas DataFrame with features as columns (wide format)
        """
        # Build MongoDB aggregation pipeline
        # Only select the fields we need: timestamp, features, feature_names (first doc only)
        pipeline = [
            {"$match": {
                "timestamp": {
                    "$gte": {"$date": format_timestamp_for_mongodb(start_hour)},
                    "$lt": {"$date": format_timestamp_for_mongodb(end_hour)}
                }
            }},
            {"$sort": {"timestamp": 1}},
            {"$addFields": {
                "timestamp_str": {
                    "$dateToString": {
                        "format": "%Y-%m-%dT%H:%M:%S.%LZ",
                        "date": "$timestamp"
                    }
                }
            }},
            # Only keep features, feature_names, and timestamp_str
            # Explicitly exclude all metadata fields
            {"$project": {
                "features": 1,
                "feature_names": 1,
                "timestamp_str": 1,
                "_id": 0
            }}
        ]
        
        # Load from MongoDB using Spark
        from src.utils.database import read_from_mongodb
        
        hour_df = read_from_mongodb(
            self.spark,
            self.db_name,
            collection_name,
            pipeline=pipeline
        )
        
        # Convert to pandas
        hour_pandas = hour_df.toPandas()
        
        if hour_pandas.empty:
            return pd.DataFrame()
        
        # Parse timestamp_str to datetime
        if 'timestamp_str' in hour_pandas.columns:
            hour_pandas['timestamp'] = pd.to_datetime(
                hour_pandas['timestamp_str'],
                format='%Y-%m-%dT%H:%M:%S.%fZ',
                utc=True
            ).dt.tz_localize(None)  # Make naive
        
        # Extract feature names (should be same for all rows, take from first)
        if 'feature_names' in hour_pandas.columns and len(hour_pandas) > 0:
            feature_names = hour_pandas['feature_names'].iloc[0]
            
            # Cache feature names for validation
            if self.feature_names_cache is None:
                self.feature_names_cache = feature_names
                logger(f"      Cached {len(feature_names)} feature names", "DEBUG")
            elif list(feature_names) != list(self.feature_names_cache):
                logger(f"      WARNING: Feature names changed!", "WARNING")
        else:
            logger(f"      WARNING: No feature_names found in data", "WARNING")
            return pd.DataFrame()
        
        # Convert features array to columns
        if 'features' in hour_pandas.columns and len(hour_pandas) > 0:
            # Extract features array into separate columns
            features_df = pd.DataFrame(
                hour_pandas['features'].tolist(),
                columns=feature_names
            )
            
            # Combine with timestamp
            result_df = pd.concat([
                hour_pandas[['timestamp']].reset_index(drop=True),
                features_df
            ], axis=1)
            
            logger(f"      Extracted {len(result_df)} samples with {len(feature_names)} features", "DEBUG")
            
            return result_df
        else:
            logger(f"      WARNING: No features array found in data", "WARNING")
            return pd.DataFrame()
    
    def extract_window_streaming(
        self, 
        collection_name: str, 
        window: Dict
    ) -> pd.DataFrame:
        """
        Extract window data hour-by-hour, then combine.
        
        Process:
        1. Identify hours covered by window
        2. Load each hour batch (extracting features array)
        3. Filter to window timestamp range
        4. Combine all hours
        
        Args:
            collection_name: MongoDB collection name (e.g., 'split_0_input')
            window: Window metadata dict with start/end timestamps
            
        Returns:
            DataFrame with window data (features as columns), or empty DataFrame if no data
        """
        window_start = parse_timestamp(window['window_start_ts'])
        window_end = parse_timestamp(window['window_end_ts'])
        expected_samples = window.get('window_n_samples', 0)
        
        # Get all hours that overlap with window
        hours = self._get_hours_in_window(window_start, window_end)
        
        logger(f"    Window spans {len(hours)} hour(s)", "DEBUG")
        
        window_data = []
        total_loaded = 0
        
        for start_hour in hours:
            end_hour = start_hour + timedelta(hours=1)
            
            try:
                # Load hour batch (features extracted to columns)
                hour_pandas = self._load_hour_batch(
                    collection_name,
                    start_hour,
                    end_hour
                )
                
                total_loaded += len(hour_pandas)
                
                # Filter to window boundaries
                if not hour_pandas.empty and 'timestamp' in hour_pandas.columns:
                    mask = (
                        (hour_pandas['timestamp'] >= window_start) & 
                        (hour_pandas['timestamp'] < window_end)
                    )
                    
                    filtered = hour_pandas[mask]
                    
                    if not filtered.empty:
                        window_data.append(filtered)
                        logger(f"      Hour {start_hour.strftime('%H:00')}: "
                              f"{len(filtered)} samples in window", "DEBUG")
            
            except Exception as e:
                logger(f"      Error loading hour {start_hour}: {e}", "WARNING")
                import traceback
                logger(traceback.format_exc(), "DEBUG")
                continue
        
        # Combine all hours for this window
        if window_data:
            full_window = pd.concat(window_data, ignore_index=True)
            full_window = full_window.sort_values('timestamp').reset_index(drop=True)
            
            # Validate sample count
            actual_samples = len(full_window)
            if expected_samples > 0 and actual_samples != expected_samples:
                logger(f"    Sample count mismatch: expected {expected_samples}, "
                      f"got {actual_samples}", "WARNING")
            
            # Log feature extraction summary
            feature_cols = [c for c in full_window.columns if c != 'timestamp']
            logger(f"    Extracted {len(full_window)} samples × {len(feature_cols)} features", "INFO")
            
            return full_window
        
        logger(f"    No data found in window", "WARNING")
        return pd.DataFrame()
    
    def extract_all_windows_streaming(
        self,
        collection_name: str,
        windows: List[Dict],
        verbose: bool = True
    ) -> List[tuple]:
        """
        Extract all windows for a split, one at a time.
        
        This is a generator-like pattern that yields (window_metadata, dataframe)
        pairs one at a time to minimize memory usage.
        
        Args:
            collection_name: MongoDB collection name
            windows: List of window metadata dicts
            verbose: Whether to log progress
            
        Yields:
            Tuples of (window_metadata, dataframe)
        """
        if verbose:
            logger(f"Extracting {len(windows)} windows from {collection_name}", "INFO")
        
        for i, window in enumerate(windows, 1):
            fold_id = window['fold_id']
            fold_type = window['fold_type']
            
            if verbose:
                logger(f"  Window {i}/{len(windows)}: "
                      f"Fold {fold_id} ({fold_type})", "INFO")
            
            # Extract window data (streaming, hour-by-hour)
            window_df = self.extract_window_streaming(collection_name, window)
            
            if not window_df.empty:
                if verbose:
                    feature_cols = [c for c in window_df.columns if c != 'timestamp']
                    logger(f"    Loaded {len(window_df)} samples × {len(feature_cols)} features", "INFO")
                
                yield (window, window_df)
            else:
                if verbose:
                    logger(f"    Skipped (no data)", "WARNING")
    
    def get_feature_names(self) -> List[str]:
        """
        Get cached feature names from data.
        
        Returns:
            List of feature names, or empty list if not yet loaded
        """
        return self.feature_names_cache or []