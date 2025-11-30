import time
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit

from src.utils.logging import logger
from src.utils.database import write_to_mongodb_preserve_objectid

from .batch_processor import BatchProcessor


class SplitMaterializer:
    """Materializes CPCV splits into separate collections."""
    
    def __init__(self, spark: SparkSession, db_name: str, input_collection: str,
                 config: dict):
        """
        Initialize split materializer.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            input_collection: Input collection name (from standardization stage)
            config: Configuration dictionary with:
                - max_splits: Maximum number of splits to materialize (None for all)
                - create_test_collection: Whether to create test_data collection
        """
        self.spark = spark
        self.db_name = db_name
        self.input_collection = input_collection
        self.config = config
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(spark, db_name)
        
        self.all_hours = None
        self.split_ids = None
        
        # Get MongoDB URI from Spark config
        self.mongo_uri = self.spark.sparkContext.getConf().get(
            'spark.mongodb.read.connection.uri', 
            'mongodb://127.0.0.1:27017/'
        )
    
    def discover_splits(self):
        """Discover all split IDs from the input collection."""
        logger('Discovering split IDs from input collection...', "INFO")
        
        # Load one document to check split_roles keys
        sample_df = (
            self.spark.read.format("mongodb")
            .option("database", self.db_name)
            .option("collection", self.input_collection)
            .load()
            .limit(1)
        )
        
        if sample_df.count() == 0:
            raise ValueError(f'Input collection {self.input_collection} is empty!')
        
        # Get split IDs from split_roles struct field names
        split_roles_schema = sample_df.schema["split_roles"].dataType
        split_ids = sorted([int(field.name) for field in split_roles_schema.fields])

        logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")

        # Apply max_splits limit if configured
        max_splits = self.config.get('max_splits', None)
        if max_splits is not None:
            split_ids = split_ids[:max_splits]
            logger(f'Limiting to first {max_splits} split(s): {split_ids}', "INFO")

        self.split_ids = split_ids
    
    def discover_hours(self):
        """Discover all processable hours."""
        self.all_hours = self.batch_processor.get_all_hours(self.input_collection)
        
        if not self.all_hours:
            raise ValueError('No processable hours found!')
    
    def materialize_all_splits(self):
        """Materialize all splits by processing hourly batches."""
        logger(f'Materializing {len(self.split_ids)} splits across {len(self.all_hours)} hours', "INFO")
        logger('ObjectId preservation: ENABLED', "INFO")
        logger('Temporal ordering preservation: ENABLED', "INFO")
        
        # Statistics tracking
        split_stats = {split_id: 0 for split_id in self.split_ids}
        test_count = 0
        total_time = 0
        
        # Process each hour
        for i, hour_str in enumerate(self.all_hours):
            batch_start = time.time()
            
            logger(f'Processing hour {i+1}/{len(self.all_hours)} - {hour_str}', "INFO")
            
            # Load hour batch
            hour_batch = self.batch_processor.load_hour_batch(self.input_collection, hour_str)
            batch_count = hour_batch.count()
            logger(f'Loaded {batch_count:,} documents', "INFO")
            
            if batch_count == 0:
                logger('Skipping empty batch', "WARNING")
                continue
            
            # Materialize each split for this hour
            hour_writes = 0
            splits_with_data = []
            for split_id in self.split_ids:
                split_df = self._extract_split(hour_batch, split_id)
                split_count = split_df.count()

                # Log split 0 count to diagnose the issue
                if split_id == 0:
                    logger(f'    [Split 0 diagnostic] After filter & dedup: {split_count} docs', "INFO")

                if split_count > 0:
                    collection_name = f"split_{split_id}_input"
                    self._write_to_collection(split_df, collection_name)
                    split_stats[split_id] += split_count
                    hour_writes += split_count
                    splits_with_data.append(f"{split_id}({split_count})")

            # Log which splits received data
            if splits_with_data:
                logger(f'  Splits written: {", ".join(splits_with_data[:5])}{"..." if len(splits_with_data) > 5 else ""}', "INFO")

            # Materialize test collection if configured
            if self.config.get('create_test_collection', False):
                # For first 3 hours, show role distribution to understand test sample presence
                if i < 3:
                    split_0_key = str(self.split_ids[0])
                    role_dist = hour_batch.groupBy(col("split_roles").getField(split_0_key)).count().collect()
                    role_info = {row[f'split_roles.{split_0_key}']: row['count'] for row in role_dist}
                    logger(f'  [Hour {i+1}] Split 0 role distribution: {role_info}', "INFO")

                test_df = self._extract_test_samples(hour_batch)
                test_batch_count = test_df.count()

                # Log test sample count for first few hours or when found
                if i < 3 or test_batch_count > 0:
                    logger(f'  Test samples in this hour: {test_batch_count}', "INFO")

                if test_batch_count > 0:
                    self._write_to_collection(test_df, "test_data")
                    test_count += test_batch_count
                    hour_writes += test_batch_count

            # Log write summary for this hour
            if hour_writes > 0:
                logger(f'  → Wrote {hour_writes:,} documents across all splits', "INFO")
            
            # Clean up
            hour_batch.unpersist()
            
            # Log progress
            batch_duration = time.time() - batch_start
            total_time += batch_duration
            avg_time = total_time / (i + 1)
            eta = avg_time * (len(self.all_hours) - i - 1)
            
            logger(f'Hour {i+1}/{len(self.all_hours)} completed in {batch_duration:.2f}s', "INFO")
            logger(f'ETA: {eta:.2f}s', "INFO")
        
        # Log final statistics
        logger('=' * 60, "INFO")
        logger('MATERIALIZATION COMPLETE', "INFO")
        logger('=' * 60, "INFO")
        
        for split_id in self.split_ids:
            logger(f'split_{split_id}_input: {split_stats[split_id]:,} documents', "INFO")
        
        if self.config.get('create_test_collection', False):
            logger(f'test_data: {test_count:,} documents', "INFO")
        
        logger(f'Total time: {total_time:.2f}s', "INFO")
    
    def _extract_split(self, df: DataFrame, split_id: int) -> DataFrame:
        """
        Extract documents for a specific split.
        Includes only train, train_warmup, and validation samples.
        Test samples are excluded - they only go to the test_data collection.

        Args:
            df: Input DataFrame with split_roles
            split_id: Split ID to extract

        Returns:
            DataFrame with non-test documents, with role field added
        """
        # Extract role for this split from split_roles struct
        split_key = str(split_id)

        # Add role column - every document gets its role for this split
        df_split = df.withColumn("role", col("split_roles").getField(split_key))

        # Debug: count before filtering (only log for first split to avoid spam)
        if split_id == 0:
            before_count = df_split.count()
            role_counts = df_split.groupBy("role").count().collect()
            role_dist = {row['role']: row['count'] for row in role_counts}
            logger(f'    [Split 0 diagnostic] Before filter: {before_count} docs, roles: {role_dist}', "INFO")

            # Check actual role values
            sample_roles = df_split.select("role").limit(5).collect()
            logger(f'    [Split 0 diagnostic] Sample role values: {[row.role for row in sample_roles]}', "INFO")

        # Keep existing fold_id and fold_type from input, drop split_roles
        # IMPORTANT: Do this BEFORE filtering to avoid Spark evaluation issues
        df_split = df_split.drop("split_roles")

        # Filter out test samples - they should only be in test_data collection
        # Use explicit filter to avoid lazy evaluation issues
        from pyspark.sql.functions import when
        df_split = df_split.filter(
            (col("role") == "train") |
            (col("role") == "train_warmup") |
            (col("role") == "validation")
        )

        # Debug: count after filter for split 0
        if split_id == 0:
            after_filter = df_split.count()
            logger(f'    [Split 0 diagnostic] After filter: {after_filter} docs', "INFO")

        # Remove duplicates - same timestamp should not appear twice
        # This ensures each split has unique timestamps only
        df_split = df_split.dropDuplicates(["timestamp"])

        # Debug: count after deduplication for split 0
        if split_id == 0:
            after_dedup = df_split.count()
            logger(f'    [Split 0 diagnostic] After dedup: {after_dedup} docs', "INFO")

        return df_split
    
    def _extract_test_samples(self, df: DataFrame) -> DataFrame:
        """
        Extract all test samples (where role='test' in ANY split).
        
        Args:
            df: Input DataFrame with split_roles
            
        Returns:
            DataFrame with test samples only
        """
        from pyspark.sql.functions import when
        
        # Get all split_roles field names
        split_roles_fields = df.schema["split_roles"].dataType.fieldNames()
        
        # Build condition to check if ANY split has role='test'
        test_conditions = [col("split_roles").getField(field) == "test" for field in split_roles_fields]
        
        # Combine all conditions with OR
        test_condition = test_conditions[0]
        for condition in test_conditions[1:]:
            test_condition = test_condition | condition
        
        # Filter to test samples
        df_test = df.filter(test_condition)
        
        # Add role field
        df_test = df_test.withColumn("role", lit("test"))
        
        # Drop split_roles but keep fold_id and fold_type
        df_test = df_test.drop("split_roles")
        
        # Remove duplicates (same sample might be test in multiple splits)
        df_test = df_test.dropDuplicates(["timestamp"])
        
        return df_test
    
    def _write_to_collection(self, df: DataFrame, collection_name: str):
        """
        Write DataFrame to MongoDB collection with ObjectId and timestamp preservation.

        Args:
            df: DataFrame to write (must have timestamp_str column)
            collection_name: Target collection name
        """
        doc_count = df.count()

        # Sort by timestamp for proper ordering
        df = df.orderBy("timestamp")

        # Apply feature projection if configured
        if hasattr(self, 'projected_features') and hasattr(self, 'apply_projection_func'):
            df = self.apply_projection_func(df, self.projected_features)

        # Write using ObjectId-preserving function
        logger(f'    Writing {doc_count:,} docs to {collection_name}...', "INFO")
        write_to_mongodb_preserve_objectid(
            df=df,
            database=self.db_name,
            collection=collection_name,
            mongo_uri=self.mongo_uri,
            mode="append"
        )
        logger(f'    ✓ Written to {collection_name}', "INFO")
    
    def get_split_collections(self) -> list:
        """
        Get list of materialized split collection names.
        
        Returns:
            List of collection names
        """
        collections = [f"split_{split_id}_input" for split_id in self.split_ids]
        
        if self.config.get('create_test_collection', False):
            collections.append("test_data")
        
        return collections