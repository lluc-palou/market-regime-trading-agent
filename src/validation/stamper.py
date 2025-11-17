from typing import List, Dict
from pyspark.sql import DataFrame
from datetime import datetime, timedelta
from pyspark.sql.functions import col, udf
from bson import ObjectId
from src.utils import logger, format_timestamp_for_mongodb
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, MapType

class DataStamper:
    """
    Stamps dataset samples with ALL role assignments using timestamp-based matching.
    
    MODIFICATIONS:
    1. Preserves ObjectId format in output (converts String → ObjectId before writing)
    2. Ensures temporal ordering with ordered writes and sequential processing
    """
    
    def __init__(self, metadata: Dict, folds: List, spark, db_name: str):
        """
        Initializes data stamper.
        """
        self.metadata = metadata
        self.folds = folds
        self.folds_by_id = {f.fold_id: f for f in folds}
        self.spark = spark
        self.db_name = db_name
    
    def stamp_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Stamps dataset samples with COMPLETE split role information using normalized timestamp strings.
        """
        metadata_broadcast = self.spark.sparkContext.broadcast(self.metadata)
        folds_broadcast = self.spark.sparkContext.broadcast([f.to_dict() for f in self.folds])
        
        def assign_complete_roles(timestamp_str):
            """
            Assigns COMPLETE role information for a given timestamp string.
            
            NOTE: This function is self-contained (no external imports) because
            it's serialized and sent to Spark workers.
            """
            # Parse timestamp string manually (no imports from src.utils)
            def parse_ts(ts_str):
                """Parse timestamp string to naive datetime."""
                if isinstance(ts_str, datetime):
                    # Already a datetime, make it naive if needed
                    if ts_str.tzinfo is not None:
                        return ts_str.replace(tzinfo=None)
                    return ts_str
                # String - remove timezone markers and parse
                ts_clean = str(ts_str).replace('Z', '').replace('+00:00', '')
                return datetime.fromisoformat(ts_clean)
            
            timestamp = parse_ts(timestamp_str)
            
            folds_data = folds_broadcast.value
            fold_id = None
            fold_type = None
            
            # Find which fold this timestamp belongs to
            for fold_dict in folds_data:
                start_ts = parse_ts(fold_dict['start_ts'])
                end_ts = parse_ts(fold_dict['end_ts'])
                
                if start_ts <= timestamp < end_ts:
                    fold_id = fold_dict['fold_id']
                    fold_type = fold_dict['fold_type']
                    break
            
            if fold_id is None:
                return (-1, 'excluded', {})
            
            # Initialize roles dictionary for ALL CPCV splits
            roles = {}
            metadata = metadata_broadcast.value
            
            def in_ranges(ranges_list):
                """Check if timestamp falls in any of the ranges."""
                for range_pair in ranges_list:
                    range_start = parse_ts(range_pair[0])
                    range_end = parse_ts(range_pair[1])
                    if range_start <= timestamp < range_end:
                        return True
                return False
            
            # Assign roles for ALL CPCV splits based on fold_type
            for split in metadata['cpcv_splits']:
                split_id = split['split_id']
                
                if fold_type == 'train':
                    # For training folds, check purge/embargo ranges
                    purged_ranges_dict = split.get('purged_ranges', {})
                    embargoed_ranges_dict = split.get('embargoed_ranges', {})
                    fold_id_str = str(fold_id)
                    
                    # Check if this sample falls in purged ranges
                    is_purged = False
                    if fold_id_str in purged_ranges_dict:
                        is_purged = in_ranges(purged_ranges_dict[fold_id_str])
                    elif fold_id in purged_ranges_dict:
                        is_purged = in_ranges(purged_ranges_dict[fold_id])
                    
                    # Check if this sample falls in embargoed ranges
                    is_embargoed = False
                    if fold_id_str in embargoed_ranges_dict:
                        is_embargoed = in_ranges(embargoed_ranges_dict[fold_id_str])
                    elif fold_id in embargoed_ranges_dict:
                        is_embargoed = in_ranges(embargoed_ranges_dict[fold_id])
                    
                    # Assign role based on priority: purged > embargoed > validation > train
                    if is_purged:
                        role = 'purged'
                    elif is_embargoed:
                        role = 'embargoed'
                    elif fold_id in split['validation_folds']:
                        role = 'validation'
                    elif fold_id in split['training_folds']:
                        role = 'train'
                    else:
                        role = 'excluded'
                
                elif fold_type == 'train_warmup':
                    role = 'train_warmup'
                
                elif fold_type == 'train_test_embargo':
                    role = 'train_test_embargo'
                
                elif fold_type == 'test':
                    role = 'test'
                
                elif fold_type == 'test_horizon':
                    role = 'test_horizon'
                
                else:
                    role = 'excluded'
                
                roles[str(split_id)] = role
            
            return (fold_id, fold_type, roles)
        
        # Create UDF that accepts STRING timestamps
        assign_roles_udf = udf(assign_complete_roles, StructType([
            StructField('fold_id', IntegerType(), True),
            StructField('fold_type', StringType(), True),
            StructField('split_roles', MapType(StringType(), StringType()), True)
        ]))
        
        # Use the normalized timestamp_str column
        stamped_df = df.withColumn('role_data', assign_roles_udf(col('timestamp_str')))
        stamped_df = stamped_df.withColumn('fold_id', col('role_data.fold_id'))
        stamped_df = stamped_df.withColumn('fold_type', col('role_data.fold_type'))               
        stamped_df = stamped_df.withColumn('split_roles', col('role_data.split_roles'))
        stamped_df = stamped_df.drop('role_data')
        
        return stamped_df
    
    def process_batches(self, input_collection: str, output_collection: str):
        """
        Processes the whole dataset samples in hourly batches, stamping with role information.
        
        MODIFICATIONS:
        - Preserves ObjectId format in output
        - Ensures temporal ordering with sequential writes
        """
        logger('Processing hourly batches with complete role stamping...', level="INFO")
        logger('ObjectId preservation: ENABLED', level="INFO")
        logger('Temporal ordering preservation: ENABLED (sequential writes)', level="INFO")
        
        data_summary = self.metadata['data_summary']
        
        # Parse timestamps manually (self-contained)
        def parse_ts(ts):
            if isinstance(ts, datetime):
                return ts.replace(tzinfo=None) if ts.tzinfo else ts
            ts_clean = str(ts).replace('Z', '').replace('+00:00', '')
            return datetime.fromisoformat(ts_clean)
        
        first_timestamp = parse_ts(data_summary['timestamp_range']['start'])
        last_timestamp = parse_ts(data_summary['timestamp_range']['end'])
        current_hour = first_timestamp.replace(minute=0, second=0, microsecond=0)
        end_boundary = last_timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        logger(f'Processing from {current_hour} to {end_boundary}', level="INFO")
        
        batch_count = 0
        total_records = 0

        # Role statistics collection removed for performance (expensive groupBy operation)
        # role_stats = {
        #     'train_warmup': 0,
        #     'train': 0,
        #     'train_test_embargo': 0,
        #     'test': 0,
        #     'test_horizon': 0,
        #     'excluded': 0
        # }
        
        while current_hour < end_boundary:
            next_hour = current_hour + timedelta(hours=1)
            
            try:
                # Load hour batch with normalized timestamps
                batch_df = self._load_hour_batch(input_collection, current_hour, next_hour)
                batch_records = batch_df.count()
                
                if batch_records > 0:
                    # Stamp batch samples with role information
                    stamped_batch = self.stamp_dataframe(batch_df)

                    # Role statistics collection removed for performance
                    # This expensive groupBy().count().collect() operation was causing timeouts
                    # role_counts = stamped_batch.groupBy('fold_type').count().collect()
                    # for row in role_counts:
                    #     fold_type = row['fold_type']
                    #     count = row['count']
                    #     if fold_type in role_stats:
                    #         role_stats[fold_type] += count

                    # Save batch with ObjectId preservation and temporal ordering
                    self._save_batch(stamped_batch, output_collection)
                    
                    total_records += batch_records
                    batch_count += 1
                    
                    logger(f'Batch {batch_count} saved ({batch_records:,} records, ordered)', level="INFO")
                    
                    stamped_batch.unpersist()
                
                batch_df.unpersist()
                
            except Exception as e:
                logger(f'Error processing batch {current_hour}: {str(e)}', level="ERROR")
                import traceback
                traceback.print_exc()
            
            current_hour = next_hour

        # Role statistics logging removed (statistics collection was removed for performance)
        # logger(f'Fold-Level Role Statistics:', level="INFO")
        # logger(f'  Train warmup: {role_stats["train_warmup"]:,} samples', level="INFO")
        # logger(f'  Train: {role_stats["train"]:,} samples', level="INFO")
        # logger(f'  Train-test embargo: {role_stats["train_test_embargo"]:,} samples', level="INFO")
        # logger(f'  Test: {role_stats["test"]:,} samples', level="INFO")
        # logger(f'  Test horizon: {role_stats["test_horizon"]:,} samples', level="INFO")
        # logger(f'  Excluded: {role_stats["excluded"]:,} samples', level="INFO")
        # logger(f'  Total: {total_records:,} samples', level="INFO")

        logger(f'Processed {batch_count} batches, {total_records:,} total records', level="INFO")
        logger(f'Output collection temporal ordering: GUARANTEED', level="INFO")
    
    def _load_hour_batch(self, collection: str, start_hour: datetime, end_hour: datetime) -> DataFrame:
        """
        Loads dataset samples by hour batches WITH NORMALIZED TIMESTAMP STRINGS.
        """
        pipeline = [
            {"$match": {
                "timestamp": {
                    "$gte": {"$date": format_timestamp_for_mongodb(start_hour)},
                    "$lt": {"$date": format_timestamp_for_mongodb(end_hour)}
                }
            }},
            {"$sort": {"timestamp": 1}},
            {"$addFields": {
                "timestamp_str": {"$dateToString": {"format": "%Y-%m-%dT%H:%M:%S.%LZ", "date": "$timestamp"}},
                "_id_str": {"$toString": "$_id"}  # Preserve _id as string for processing
            }}
        ]
        
        batch_df = (
            self.spark.read.format("mongodb")
            .option("database", self.db_name)
            .option("collection", collection)
            .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
            .load()
        )
        
        return batch_df
    
    def _save_batch(self, batch_df: DataFrame, output_collection: str):
        """
        Saves the batch to output collection with ObjectId preservation and temporal ordering.

        CRITICAL MODIFICATIONS:
        1. Converts _id from String back to ObjectId before writing
        2. Uses timestamp_str (UTC string) instead of Spark timestamp to avoid timezone conversion
        3. Uses ordered=True to preserve write order
        4. Uses toLocalIterator() to stream rows instead of collect() for large batches

        This guarantees:
        - ObjectId format is maintained in output collection
        - Timestamps remain in original UTC timezone (no +2h shift)
        - Temporal ordering is preserved across all batches
        - Memory efficient processing for large batches
        """
        # Use toLocalIterator to stream rows instead of collecting all at once
        # This avoids loading the entire batch into driver memory

        # Convert to list of dicts and fix ObjectId + timestamps
        documents = []
        for row in batch_df.toLocalIterator():
            doc = row.asDict()

            # CRITICAL FIX: Use timestamp_str to reconstruct UTC naive datetime
            # This avoids Spark's timezone conversion issues
            if 'timestamp_str' in doc:
                ts_str = doc['timestamp_str']
                # Parse the UTC string (format: "2025-07-04T00:00:13.211Z")
                ts_str_clean = ts_str.replace('Z', '').replace('+00:00', '')
                doc['timestamp'] = datetime.fromisoformat(ts_str_clean)

            # Remove temporary processing fields
            doc.pop('timestamp_str', None)
            doc.pop('_id_str', None)
            
            # CRITICAL FIX: Convert _id from String back to ObjectId
            if '_id' in doc and isinstance(doc['_id'], str):
                try:
                    doc['_id'] = ObjectId(doc['_id'])
                except Exception as e:
                    logger(f'Warning: Could not convert _id to ObjectId: {doc["_id"]}', level="WARNING")

            documents.append(doc)

            # Write in chunks to avoid memory buildup (every 10,000 documents)
            if len(documents) >= 10000:
                self._write_documents_to_mongo(documents, output_collection)
                documents = []

        # Write any remaining documents
        if documents:
            self._write_documents_to_mongo(documents, output_collection)

    def _write_documents_to_mongo(self, documents: list, output_collection: str):
        """
        Helper method to write a list of documents to MongoDB.
        Extracted to allow chunked writing for large batches.
        """
        from pymongo import MongoClient

        # Get MongoDB URI from Spark config or use default
        mongo_uri = self.spark.sparkContext.getConf().get(
            'spark.mongodb.read.connection.uri',
            'mongodb://127.0.0.1:27017/'
        )

        client = MongoClient(mongo_uri)
        db = client[self.db_name]
        collection = db[output_collection]

        # Insert with ordered=True to preserve temporal ordering
        try:
            collection.insert_many(documents, ordered=True)
        except Exception as e:
            logger(f'Error inserting batch: {str(e)}', level="ERROR")
            raise
        finally:
            client.close()