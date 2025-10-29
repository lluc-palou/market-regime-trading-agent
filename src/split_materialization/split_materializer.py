import time
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit

from src.utils.logging import logger
from src.utils.database import write_to_mongodb

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
        
        # Apply max_splits limit
        max_splits = self.config.get('max_splits')
        if max_splits is not None:
            split_ids = split_ids[:max_splits]
            logger(f'Limiting to first {max_splits} splits: {split_ids}', "INFO")
        
        self.split_ids = split_ids
    
    def discover_hours(self):
        """Discover all processable hours."""
        self.all_hours = self.batch_processor.get_all_hours(self.input_collection)
        
        if not self.all_hours:
            raise ValueError('No processable hours found!')
    
    def materialize_all_splits(self):
        """Materialize all splits by processing hourly batches."""
        logger(f'Materializing {len(self.split_ids)} splits across {len(self.all_hours)} hours', "INFO")
        
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
            for split_id in self.split_ids:
                split_df = self._extract_split(hour_batch, split_id)
                split_count = split_df.count()
                
                if split_count > 0:
                    collection_name = f"split_{split_id}_input"
                    self._write_to_collection(split_df, collection_name)
                    split_stats[split_id] += split_count
            
            # Materialize test collection if configured
            if self.config.get('create_test_collection', False):
                test_df = self._extract_test_samples(hour_batch)
                test_batch_count = test_df.count()
                
                if test_batch_count > 0:
                    self._write_to_collection(test_df, "test_data")
                    test_count += test_batch_count
            
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
            logger(f'split_{split_id}: {split_stats[split_id]:,} documents', "INFO")
        
        if self.config.get('create_test_collection', False):
            logger(f'test_data: {test_count:,} documents', "INFO")
        
        logger(f'Total time: {total_time:.2f}s', "INFO")
    
    def _extract_split(self, df: DataFrame, split_id: int) -> DataFrame:
        """
        Extract documents for a specific split.
        Every document is included with its role for this split.
        
        Args:
            df: Input DataFrame with split_roles
            split_id: Split ID to extract
            
        Returns:
            DataFrame with all documents, with role field added
        """
        # Extract role for this split from split_roles struct
        split_key = str(split_id)
        
        # Add role column - every document gets its role for this split
        df_split = df.withColumn("role", col("split_roles").getField(split_key))
        
        # Keep existing fold_id and fold_type from input, drop split_roles
        df_split = df_split.drop("split_roles")
        
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
        Write DataFrame to MongoDB collection.
        
        Args:
            df: DataFrame to write
            collection_name: Target collection name
        """
        # Drop timestamp_str (only used for processing), keep timestamp as-is
        if "timestamp_str" in df.columns:
            df = df.drop("timestamp_str")
        
        # Sort by timestamp for proper ordering
        df = df.orderBy("timestamp")
        
        # Write to MongoDB (timestamp is already in correct format)
        write_to_mongodb(
            df,
            self.db_name,
            collection_name,
            mode="append"
        )
    
    def get_split_collections(self) -> list:
        """
        Get list of materialized split collection names.
        
        Returns:
            List of collection names
        """
        collections = [f"split_{split_id}" for split_id in self.split_ids]
        
        if self.config.get('create_test_collection', False):
            collections.append("test_data")
        
        return collections