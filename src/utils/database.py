from pyspark.sql.functions import expr
from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

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

def write_with_timestamp_conversion(
    df: DataFrame,
    database: str,
    collection: str,
    timestamp_col: str = "timestamp",
    mode: str = "append"
) -> None:
    """
    Writes DataFrame to MongoDB, converting timestamp string back to timestamp type.
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