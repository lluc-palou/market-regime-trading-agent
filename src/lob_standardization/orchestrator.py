"""
Standardization Orchestrator Module

Main orchestrator for the LOB standardization pipeline.
Includes feature consolidation into array format.
"""

import time
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import expr, col, array, lit, struct

from src.utils.logging import logger
from src.utils.database import read_all_with_timestamp_strings, write_with_timestamp_conversion

from .price_standardizer import PriceStandardizer
from .volume_quantizer import VolumeQuantizer
from .volume_coverage_analyzer import CoverageAnalyzer
from .batch_processor import BatchProcessor


class StandardizationOrchestrator:
    """Orchestrates the LOB standardization pipeline."""
    
    def __init__(self, spark: SparkSession, db_name: str, input_collection: str,
                 output_collection: str, config: dict):
        """
        Initialize standardization orchestrator.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            input_collection: Input collection name (features_lob)
            output_collection: Output collection name (standard_lob)
            config: Configuration dictionary with:
                - B: number of bins
                - delta: clipping threshold
                - epsilon: minimum spacing
                - consolidate_features: whether to consolidate features (default: True)
        """
        self.spark = spark
        self.db_name = db_name
        self.input_collection = input_collection
        self.output_collection = output_collection
        self.config = config
        
        # Initialize components
        self.price_standardizer = PriceStandardizer(
            eps=config.get('eps', 1e-8),
            min_denom=config.get('min_denom', 1e-6)
        )
        self.volume_quantizer = VolumeQuantizer(
            spark=spark,
            B=config['B'],
            delta=config['delta'],
            epsilon=config['epsilon']
        )
        self.coverage_analyzer = CoverageAnalyzer()
        self.batch_processor = BatchProcessor(config['required_past_hours'])
        
        self.features_lob = None
        self.processable_hours = None
        self.feature_order = None  # Will store feature column order
    
    def load_raw_data(self):
        """Discover available hours from MongoDB (lightweight operation)."""
        logger(f'Discovering available hours from {self.input_collection}', "INFO")

        # Load ONLY timestamps to discover hours (not all data)
        # This is a lightweight query that uses the timestamp index
        from src.utils.database import read_sorted_with_timestamp_strings
        timestamps_only = read_sorted_with_timestamp_strings(
            self.spark,
            self.db_name,
            self.input_collection,
            additional_fields=[]  # No additional fields needed
        )

        count = timestamps_only.count()
        num_cols = len(timestamps_only.columns)
        logger(f'Found {count:,} records', "INFO")

        # Cache just the timestamps for hour discovery
        self.features_lob = timestamps_only

    def determine_processable_hours(self):
        """Determine which hours can be processed."""
        self.processable_hours = self.batch_processor.get_processable_hours(self.features_lob)

        if not self.processable_hours:
            raise ValueError('No processable hours found!')

        # Can unpersist timestamps now that we have the hour list
        self.features_lob.unpersist()
        self.features_lob = None
    
    def process_all_batches(self):
        """Process all hourly batches."""
        volume_coverage_analysis = self.config.get('volume_coverage_analysis', False)

        logger(f'Processing {len(self.processable_hours)} hours', "INFO")
        logger(f'Pipeline mode: {"VOLUME COVERAGE ANALYSIS" if volume_coverage_analysis else "MODELING"}', "INFO")

        # Import needed for MongoDB batch loading
        from src.utils.database import read_all_with_timestamp_strings
        from src.utils.timestamp import parse_hour_string, add_hours, format_timestamp_for_mongodb

        total_processed = 0
        total_time = 0

        for i, target_hour_str in enumerate(self.processable_hours):
            batch_start = time.time()
            is_first = (i == 0)

            logger(f'Processing {i+1}/{len(self.processable_hours)} - {target_hour_str}', "INFO")

            # Calculate time window with context
            target_dt = parse_hour_string(target_hour_str)
            past_start_dt = add_hours(target_dt, -self.batch_processor.required_past_hours)
            future_end_dt = add_hours(target_dt, 1)  # Just the target hour

            # Format for MongoDB query
            past_start_str = format_timestamp_for_mongodb(past_start_dt)
            future_end_str = format_timestamp_for_mongodb(future_end_dt)

            # Build MongoDB pipeline with $match for efficient timestamp filtering
            pipeline = [
                {"$match": {
                    "timestamp": {
                        "$gte": {"$date": past_start_str},
                        "$lt": {"$date": future_end_str}
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
                }}
            ]

            # CRITICAL: Load batch directly from MongoDB using timestamp index
            from src.utils.database import read_from_mongodb
            hour_batch = read_from_mongodb(
                self.spark,
                self.db_name,
                self.input_collection,
                pipeline
            )

            batch_count = hour_batch.count()
            logger(f'Loaded {batch_count:,} rows from MongoDB (range: {past_start_str} to {future_end_str})', "INFO")

            if batch_count > 0:
                # Process batch
                standardized_df = self._process_batch(hour_batch, target_hour_str, is_first, volume_coverage_analysis)

                # Write to database
                self._write_to_database(standardized_df)

                processed_count = standardized_df.count()
                total_processed += processed_count
                standardized_df.unpersist()
            else:
                logger(f'Skipping empty batch', "WARNING")

            # Statistics
            batch_duration = time.time() - batch_start
            total_time += batch_duration
            avg_time = total_time / (i + 1)
            eta = avg_time * (len(self.processable_hours) - i - 1)

            logger(f'Batch completed in {batch_duration:.2f}s', "INFO")
            logger(f'Progress: {i+1}/{len(self.processable_hours)}, ETA: {eta:.2f}s', "INFO")

            hour_batch.unpersist()
        
        logger(f'Total processed: {total_processed:,} rows in {total_time:.2f}s', "INFO")
        
        # Log feature order for reference
        if self.feature_order:
            logger(f'Feature order saved: {len(self.feature_order)} features', "INFO")
            logger(f'First 10 features: {self.feature_order[:10]}', "INFO")
    
    def _identify_feature_columns(self, df: DataFrame) -> tuple:
        """
        Identify and categorize feature columns.
        
        Returns:
            Tuple of (target_features, input_features, all_features)
            - target_features: Forward returns (targets)
            - input_features: All other features (for training)
            - all_features: All feature columns in order
        """
        all_cols = df.columns
        
        # Identify feature columns by prefix
        fwd_logret_cols = sorted([c for c in all_cols if c.startswith("fwd_logret_")])
        past_logret_cols = sorted([c for c in all_cols if c.startswith("past_logret_")])
        depth_imbalance_cols = sorted([c for c in all_cols if c.startswith("depth_imbalance_")])
        depth_cols = sorted([c for c in all_cols if c.startswith("depth_") and not c.startswith("depth_imbalance_")])
        liquidity_cols = sorted([c for c in all_cols if c.startswith("liquidity_")])
        price_impact_cols = sorted([c for c in all_cols if c.startswith("price_impact_")])
        
        # Other scalar features
        other_features = [c for c in ["volatility", "spread", "microprice"] if c in all_cols]
        
        # Define feature order (targets first, then inputs)
        target_features = fwd_logret_cols  # These are targets
        input_features = (
            past_logret_cols + 
            other_features + 
            depth_imbalance_cols + 
            depth_cols + 
            liquidity_cols + 
            price_impact_cols
        )
        
        # All features in order
        all_features = target_features + input_features
        
        return target_features, input_features, all_features
    
    def _consolidate_features(self, df: DataFrame, all_feature_columns: list) -> DataFrame:
        """
        Convert individual feature columns into array format.
        
        Creates:
        - feature_names: Array of feature names (for reference)
        - features: Array of feature values (in same order)
        
        Args:
            df: DataFrame with individual feature columns
            all_feature_columns: Ordered list of feature column names
            
        Returns:
            DataFrame with consolidated features
        """
        logger('Consolidating features into array format...', "INFO")
        
        # Create feature_names array (constant for all rows)
        df = df.withColumn(
            "feature_names",
            array(*[lit(name) for name in all_feature_columns])
        )
        
        # Create features array (values in same order)
        df = df.withColumn(
            "features",
            array(*[col(name) for name in all_feature_columns])
        )
        
        # Drop individual feature columns
        df = df.drop(*all_feature_columns)
        
        logger(f'Consolidated {len(all_feature_columns)} features into array', "INFO")
        
        return df
    
    def _process_batch(self, batch: DataFrame, target_hour_str: str, 
                       is_first_batch: bool, volume_coverage_analysis: bool) -> DataFrame:
        """
        Process a single hourly batch.
        
        Args:
            batch: DataFrame with batch data (includes all features from Stage 4)
            target_hour_str: Target hour string
            is_first_batch: Whether this is the first batch
            volume_coverage_analysis: Whether to run coverage analysis mode
            
        Returns:
            DataFrame with standardized LOB data (bins + consolidated features)
        """
        logger(f'Standardizing LOB for {target_hour_str} (first={is_first_batch})', "INFO")
        
        # Prepare base dataframe - keep ALL columns
        base = (
            batch.withColumn("trading_hour_str", expr("substring(timestamp_str, 1, 13) || ':00:00.000Z'"))
            .repartition(4)
            .sortWithinPartitions("timestamp_str")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = base.count()
        
        # Identify feature columns
        target_features, input_features, all_features = self._identify_feature_columns(base)
        
        # Store feature order (first batch only)
        if is_first_batch and self.feature_order is None:
            self.feature_order = all_features
            logger(f'Feature order established: {len(all_features)} features', "INFO")
            logger(f'Targets: {len(target_features)}, Inputs: {len(input_features)}', "INFO")
        
        metadata_cols = ["timestamp_str", "trading_hour_str", "fold_type", "fold_id", "split_roles"]
        
        logger(f'Processing with {len(all_features)} feature columns', "INFO")
        
        # Step 1: Prepare ordered price-volume pairs
        logger('Step 1: Preparing ordered price-volume pairs...', "INFO")
        df = self.price_standardizer.prepare_ordered_price_volume_pairs(base)
        
        # Keep all features alongside price_volume_pairs
        features_to_keep = [c for c in metadata_cols + all_features if c in df.columns]
        features_to_keep.append("price_volume_pairs")
        
        # Add mid_price and volatility if available (from Stage 4)
        if "mid_price" in df.columns:
            features_to_keep.append("mid_price")
        if "volatility" in df.columns:
            features_to_keep.append("volatility")
        
        df = df.select(*features_to_keep)
        
        # Step 2: Verify we have mid_price and volatility from Stage 4
        logger('Step 2: Using mid_price and volatility from Stage 4 features...', "INFO")
        if "mid_price" not in df.columns or "volatility" not in df.columns:
            raise ValueError("mid_price and volatility must exist in features_lob from Stage 4!")
        
        # Step 3: Standardize prices
        logger('Step 3: Standardizing prices...', "INFO")
        df = self.price_standardizer.standardize_prices(df)
        
        if volume_coverage_analysis:
            # VOLUME COVERAGE ANALYSIS MODE
            logger('=== VOLUME COVERAGE ANALYSIS MODE ===', "INFO")
            df = self.coverage_analyzer.analyze_coverage(df, self.config['delta'])
            
            keep = [c for c in metadata_cols + all_features if c in df.columns]
            keep.extend(["total_volume_before_clip", "total_volume_after_clip", "volume_coverage_pct"])
            df = df.select(*keep)
        else:
            # MODELING MODE
            logger('=== MODELING MODE: clip -> extract -> quantize -> normalize ===', "INFO")
            
            # Step 4: Clip price levels
            logger('Step 4: Clipping price levels...', "INFO")
            df = self.price_standardizer.clip_price_levels(df, self.config['delta'])
            
            # Step 5: Extract prices and volumes
            logger('Step 5: Extracting prices and volumes...', "INFO")
            df = self.price_standardizer.extract_prices_and_volumes(df)
            
            # Step 6: Quantize and aggregate volumes
            logger('Step 6: Logarithmic quantization and volume aggregation...', "INFO")
            df = self.volume_quantizer.quantize_and_aggregate(df)
            
            # Step 7: Normalize volumes
            logger('Step 7: Normalizing binned volumes...', "INFO")
            df = self.volume_quantizer.normalize_bins(df)
            
            # Keep features + bins, drop intermediate columns
            keep = [c for c in metadata_cols + all_features if c in df.columns]
            keep.append("bins")
            df = df.select(*keep)
            
            # Step 8: Consolidate features into array
            logger('Step 8: Consolidating features into array format...', "INFO")
            df = self._consolidate_features(df, all_features)
        
        # Filter to target hour
        df_output = self.batch_processor.filter_to_target_hour(df, target_hour_str, is_first_batch)
        
        # Rename timestamp for output
        df_output = df_output.drop("trading_hour_str").withColumnRenamed("timestamp_str", "timestamp")
        
        cnt = df_output.count()
        output_cols = len(df_output.columns)
        logger(f'LOB standardization complete - {cnt:,} rows, {output_cols} columns', "INFO")
        base.unpersist()
        
        return df_output
    
    def _write_to_database(self, df: DataFrame):
        """
        Write standardized LOB to MongoDB.
        
        Args:
            df: DataFrame to write
        """
        logger('Writing standardized LOB to database...', "INFO")
        total = df.count()
        logger(f'Writing {total:,} rows...', "INFO")
        
        # Sort by timestamp for proper ordering
        df = df.orderBy("timestamp")
        
        t0 = time.time()
        write_with_timestamp_conversion(
            df,
            self.db_name,
            self.output_collection,
            timestamp_col="timestamp",
            mode="append"
        )
        
        logger(f'Write completed in {time.time() - t0:.2f}s', "INFO")
    
    def get_feature_order(self) -> list:
        """
        Get the established feature order.
        
        Returns:
            List of feature names in order, or None if not yet established
        """
        return self.feature_order