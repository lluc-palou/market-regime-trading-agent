from typing import List, Dict
from pyspark.sql import DataFrame
from datetime import datetime, timedelta
from pyspark.sql.functions import col, expr, lit, create_map
from bson import ObjectId
from src.utils import logger, format_timestamp_for_mongodb

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

    def _build_fold_assignment_sql(self) -> str:
        """
        Build SQL CASE expression to determine fold_id and fold_type from timestamp_str.

        Returns:
            SQL expression for fold assignment
        """
        cases = []
        for fold in self.folds:
            fold_dict = fold.to_dict()
            start_str = str(fold_dict['start_ts']).replace('Z', '').replace('+00:00', '')
            end_str = str(fold_dict['end_ts']).replace('Z', '').replace('+00:00', '')
            fold_id = fold_dict['fold_id']
            fold_type = fold_dict['fold_type']

            cases.append(f"""
                WHEN timestamp_str >= '{start_str}' AND timestamp_str < '{end_str}'
                THEN named_struct('fold_id', {fold_id}, 'fold_type', '{fold_type}')
            """)

        # Default case: excluded
        return f"""
            CASE
                {' '.join(cases)}
                ELSE named_struct('fold_id', -1, 'fold_type', 'excluded')
            END
        """

    def _build_role_assignment_sql(self, split_id: int, split: Dict) -> str:
        """
        Build SQL CASE expression to determine role for a specific split.

        Args:
            split_id: Split ID
            split: Split metadata dict

        Returns:
            SQL expression for role assignment
        """
        purged_ranges_dict = split.get('purged_ranges', {})
        embargoed_ranges_dict = split.get('embargoed_ranges', {})
        validation_folds = split.get('validation_folds', [])
        training_folds = split.get('training_folds', [])

        # Build purge/embargo range checks
        purge_conditions = []
        embargo_conditions = []

        for fold_id_key, ranges in purged_ranges_dict.items():
            for start, end in ranges:
                start_str = str(start).replace('Z', '').replace('+00:00', '')
                end_str = str(end).replace('Z', '').replace('+00:00', '')
                purge_conditions.append(f"(timestamp_str >= '{start_str}' AND timestamp_str < '{end_str}')")

        for fold_id_key, ranges in embargoed_ranges_dict.items():
            for start, end in ranges:
                start_str = str(start).replace('Z', '').replace('+00:00', '')
                end_str = str(end).replace('Z', '').replace('+00:00', '')
                embargo_conditions.append(f"(timestamp_str >= '{start_str}' AND timestamp_str < '{end_str}')")

        purge_check = " OR ".join(purge_conditions) if purge_conditions else "FALSE"
        embargo_check = " OR ".join(embargo_conditions) if embargo_conditions else "FALSE"

        validation_check = f"fold_info.fold_id IN ({','.join(map(str, validation_folds))})" if validation_folds else "FALSE"
        training_check = f"fold_info.fold_id IN ({','.join(map(str, training_folds))})" if training_folds else "FALSE"

        # Build role assignment logic
        return f"""
            CASE
                WHEN fold_info.fold_type = 'train' THEN
                    CASE
                        WHEN ({purge_check}) THEN 'purged'
                        WHEN ({embargo_check}) THEN 'embargoed'
                        WHEN ({validation_check}) THEN 'validation'
                        WHEN ({training_check}) THEN 'train'
                        ELSE 'excluded'
                    END
                WHEN fold_info.fold_type = 'train_warmup' THEN 'train_warmup'
                WHEN fold_info.fold_type = 'train_test_embargo' THEN 'train_test_embargo'
                WHEN fold_info.fold_type = 'test' THEN 'test'
                WHEN fold_info.fold_type = 'test_horizon' THEN 'test_horizon'
                ELSE 'excluded'
            END
        """

    def stamp_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Stamps dataset samples with COMPLETE split role information using native Spark SQL.
        No UDFs for better performance on large datasets.
        """
        logger('Stamping dataframe with native Spark SQL (no UDFs)', "INFO")

        # Step 1: Determine fold_id and fold_type from timestamp
        fold_assignment_expr = self._build_fold_assignment_sql()
        stamped_df = df.withColumn('fold_info', expr(fold_assignment_expr))
        stamped_df = stamped_df.withColumn('fold_id', col('fold_info.fold_id'))
        stamped_df = stamped_df.withColumn('fold_type', col('fold_info.fold_type'))

        # Step 2: Build split_roles map for all splits
        split_role_exprs = []
        for split in self.metadata['cpcv_splits']:
            split_id = split['split_id']
            role_expr = self._build_role_assignment_sql(split_id, split)
            split_role_exprs.append(f"'{split_id}', ({role_expr})")

        # Create map of split_id -> role
        split_roles_expr = f"map({', '.join(split_role_exprs)})"
        stamped_df = stamped_df.withColumn('split_roles', expr(split_roles_expr))

        # Drop intermediate column
        stamped_df = stamped_df.drop('fold_info')

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

            # Write in chunks to avoid socket timeouts with large documents (~200KB each)
            # Chunk size of 30 = ~6MB per write operation
            if len(documents) >= 30:
                self._write_documents_to_mongo(documents, output_collection)
                documents = []

        # Write any remaining documents
        if documents:
            self._write_documents_to_mongo(documents, output_collection)

    def _write_documents_to_mongo(self, documents: list, output_collection: str):
        """
        Helper method to write a list of documents to MongoDB.
        Extracted to allow chunked writing for large batches.

        Uses extended socket timeout for large documents (~200KB each).
        """
        from pymongo import MongoClient

        # Get MongoDB URI from Spark config or use default
        mongo_uri = self.spark.sparkContext.getConf().get(
            'spark.mongodb.read.connection.uri',
            'mongodb://127.0.0.1:27017/'
        )

        # Use extended socket timeout for large documents (2 hours = 7200000ms)
        client = MongoClient(mongo_uri, socketTimeoutMS=7200000)
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
