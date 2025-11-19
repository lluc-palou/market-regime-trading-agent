from pyspark.sql.functions import expr
from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

# =================================================================================================
# Read Operations
# =================================================================================================

def read_from_mongodb(
    spark: SparkSession,
    database: str,
    collection: str,
    pipeline: Optional[List[Dict]] = None
) -> DataFrame:
    """
    Reads data from MongoDB collection with optional aggregation pipeline.
    """
    reader = (
        spark.read.format("mongodb")
        .option("database", database)
        .option("collection", collection)
    )
    
    if pipeline:
        # Converts pipeline to JSON string format expected by MongoDB connector
        pipeline_str = str(pipeline).replace("'", '"')
        reader = reader.option("aggregation.pipeline", pipeline_str)
    
    return reader.load()

def read_sorted_with_timestamp_strings(
    spark: SparkSession,
    database: str,
    collection: str,
    additional_fields: Optional[List[str]] = None
) -> DataFrame:
    """
    Reads from MongoDB and returns timestamps as normalized strings.
    """
    # Builds projection
    projection = {
        "timestamp_str": {
            "$dateToString": {
                "format": "%Y-%m-%dT%H:%M:%S.%LZ",
                "date": "$timestamp"
            }
        }
    }

    # Adds additional fields to projection
    if additional_fields:
        for field in additional_fields:
            projection[field] = 1

    pipeline = [
        {"$sort": {"timestamp": 1}},
        {"$project": projection}
    ]

    return read_from_mongodb(spark, database, collection, pipeline)

def read_hourly_batch_from_mongodb(
    spark: SparkSession,
    database: str,
    collection: str,
    start_timestamp_str: str,
    end_timestamp_str: str,
    additional_fields: Optional[List[str]] = None
) -> DataFrame:
    """
    Reads a specific time range from MongoDB using timestamp index for efficiency.

    CRITICAL: Uses $match with timestamp range to leverage MongoDB timestamp index.
    This enables O(log N + matches) query performance instead of O(N) full scan.

    Args:
        spark: SparkSession instance
        database: Database name
        collection: Collection name
        start_timestamp_str: Start timestamp (ISO format with Z)
        end_timestamp_str: End timestamp (ISO format with Z)
        additional_fields: Additional fields to project

    Returns:
        DataFrame with documents in time range, sorted by timestamp
    """
    # Build projection
    projection = {
        "timestamp_str": {
            "$dateToString": {
                "format": "%Y-%m-%dT%H:%M:%S.%LZ",
                "date": "$timestamp"
            }
        }
    }

    # Add additional fields to projection
    if additional_fields:
        for field in additional_fields:
            projection[field] = 1

    # Pipeline with $match BEFORE $sort to use index
    pipeline = [
        {"$match": {
            "timestamp": {
                "$gte": {"$date": start_timestamp_str},
                "$lt": {"$date": end_timestamp_str}
            }
        }},
        {"$sort": {"timestamp": 1}},
        {"$project": projection}
    ]

    return read_from_mongodb(spark, database, collection, pipeline)

def read_all_with_timestamp_strings(
    spark: SparkSession,
    database: str,
    collection: str
) -> DataFrame:
    """
    Reads ALL fields from MongoDB and adds timestamp_str column.
    
    Uses $addFields to preserve ALL existing fields while adding timestamp_str.
    This is useful when you need all fields (like feature columns) instead of 
    just a subset.
    
    Args:
        spark: SparkSession instance
        database: Database name
        collection: Collection name
        
    Returns:
        DataFrame with all fields + timestamp_str
    """
    pipeline = [
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
    
    return read_from_mongodb(spark, database, collection, pipeline)

# =================================================================================================
# Write Operations
# =================================================================================================

def write_to_mongodb(
    df: DataFrame,
    database: str,
    collection: str,
    mode: str = "append",
    ordered: bool = False,
    timestamp_column: Optional[str] = None
) -> None:
    """
    Writes DataFrame to MongoDB collection.
    
    IMPORTANT: This function does NOT preserve ObjectId format.
    Use write_to_mongodb_preserve_objectid() if you need to maintain ObjectId type.
    """
    df_output = df
    
    # Converts timestamp string to proper timestamp if needed
    if timestamp_column and timestamp_column in df.columns:
        df_output = df.withColumn(
            timestamp_column,
            expr(f"to_timestamp({timestamp_column}, \"yyyy-MM-dd'T'HH:mm:ss.SSSX\")")
        )
    
    (df_output.write.format("mongodb")
     .option("database", database)
     .option("collection", collection)
     .option("ordered", str(ordered).lower())
     .mode(mode)
     .save())

def write_to_mongodb_preserve_objectid(
    df: DataFrame,
    database: str,
    collection: str,
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    mode: str = "append"
) -> None:
    """
    Writes DataFrame to MongoDB while preserving ObjectId format and timestamp UTC naive format.
    
    This function:
    1. Collects Spark DataFrame to Python
    2. Converts _id from String back to ObjectId
    3. Parses timestamp from timestamp_str (if present) to avoid timezone shifts
    4. Writes using PyMongo with ordered inserts (skips duplicates if in append mode)
    
    Args:
        df: DataFrame to write (should have timestamp_str if timestamp needs to be preserved)
        database: Database name
        collection: Collection name
        mongo_uri: MongoDB connection URI
        mode: Write mode ('append' or 'overwrite')
    
    Note: This is slower than Spark writes but guarantees data integrity.
    """
    from pymongo.errors import BulkWriteError
    
    # Collect DataFrame to Python
    batch_data = df.collect()
    
    if not batch_data:
        return
    
    # Convert to list of dicts with proper types
    documents = []
    skipped_ids = []
    
    for row in batch_data:
        doc = row.asDict()
        
        # FIX 1: Convert timestamp from timestamp_str (UTC string) to avoid timezone shifts
        if 'timestamp_str' in doc:
            ts_str = doc['timestamp_str']
            # Parse the UTC string (format: "2025-07-04T00:00:13.211Z")
            ts_str_clean = ts_str.replace('Z', '').replace('+00:00', '')
            doc['timestamp'] = datetime.fromisoformat(ts_str_clean)
        
        # Remove temporary processing fields
        doc.pop('timestamp_str', None)
        doc.pop('_id_str', None)
        
        # FIX 2: Convert _id from String back to ObjectId
        if '_id' in doc and isinstance(doc['_id'], str):
            try:
                doc['_id'] = ObjectId(doc['_id'])
            except Exception as e:
                # If conversion fails, skip this document
                print(f'Warning: Could not convert _id to ObjectId: {doc["_id"]} - {e}')
                skipped_ids.append(doc.get('_id', 'unknown'))
                continue
        
        documents.append(doc)
    
    if skipped_ids:
        print(f'Warning: Skipped {len(skipped_ids)} documents with invalid ObjectIds')
    
    if not documents:
        print('Warning: No valid documents to insert')
        return
    
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database]
    coll = db[collection]
    
    try:
        # Handle mode
        if mode == "overwrite":
            # Drop collection first
            coll.drop()
            # Insert all documents
            if documents:
                coll.insert_many(documents, ordered=True)
        else:  # append mode
            # In append mode, handle duplicate key errors gracefully
            try:
                coll.insert_many(documents, ordered=False)  # ordered=False to continue on duplicates
            except BulkWriteError as bwe:
                # Extract detailed error information
                write_errors = bwe.details.get('writeErrors', [])
                duplicate_count = sum(1 for err in write_errors if err.get('code') == 11000)
                other_errors = [err for err in write_errors if err.get('code') != 11000]
                
                # If only duplicate key errors, that's expected in append mode
                if other_errors:
                    print(f'Error: Non-duplicate write errors occurred:')
                    for err in other_errors[:5]:  # Show first 5 errors
                        print(f"  - Code {err.get('code')}: {err.get('errmsg', 'Unknown error')}")
                    raise
                else:
                    # Only duplicates - log but continue
                    inserted_count = bwe.details.get('nInserted', 0)
                    print(f'Info: Inserted {inserted_count} documents, skipped {duplicate_count} duplicates')
    
    except Exception as e:
        print(f'Error inserting documents to {database}.{collection}: {str(e)}')
        # Print first document structure for debugging
        if documents:
            print('Sample document structure:')
            sample = {k: type(v).__name__ for k, v in list(documents[0].items())[:10]}
            print(f'  {sample}')
        raise
    
    finally:
        client.close()

def write_with_timestamp_conversion(
    df: DataFrame,
    database: str,
    collection: str,
    timestamp_col: str = "timestamp",
    mode: str = "append"
) -> None:
    """
    Writes DataFrame to MongoDB, converting timestamp string back to timestamp type.
    
    IMPORTANT: This function does NOT preserve ObjectId format.
    Use write_to_mongodb_preserve_objectid() if you need to maintain ObjectId type.
    """
    df_output = df.withColumn(
        timestamp_col,
        expr(f"to_timestamp(replace({timestamp_col}, 'Z', ''), \"yyyy-MM-dd'T'HH:mm:ss.SSS\")")
    )
    
    (df_output.write.format("mongodb")
     .option("database", database)
     .option("collection", collection)
     .option("ordered", "false")
     .mode(mode)
     .save())

# =================================================================================================
# Logging & Tracking Operations
# =================================================================================================

def update_log_collection(
    spark: SparkSession,
    database: str,
    log_collection: str,
    filename: str,
    record_type: str,
    record_count: int
) -> None:
    """
    Updates a log collection to track processed files.
    """
    schema = StructType([
        StructField("filename", StringType(), False),
        StructField("type", StringType(), False),
        StructField("record_count", IntegerType(), False),
    ])
    
    log_row_df = spark.createDataFrame(
        [(filename, record_type, int(record_count))],
        schema=schema
    )
    
    write_to_mongodb(log_row_df, database, log_collection, mode="append")

def get_logged_files(
    spark: SparkSession,
    database: str,
    log_collection: str,
    record_type: str
) -> set:
    """
    Gets set of filenames that have already been logged/processed.
    """
    try:
        log_df = read_from_mongodb(spark, database, log_collection)
        
        uploaded = (
            log_df.filter(log_df["type"] == record_type)
            .select("filename")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        
        return set(uploaded)
    
    except Exception:
        # Collection doesn't exist yet
        return set()

# =================================================================================================
# Utility Operations
# =================================================================================================

def count_documents(
    spark: SparkSession,
    database: str,
    collection: str
) -> int:
    """
    Counts documents in a MongoDB collection.
    """
    try:
        df = read_from_mongodb(spark, database, collection)
        return df.count()
    except Exception:
        return 0