import os
import sys
import time
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import StructType, ArrayType, DoubleType, TimestampType

# ===== ADD THIS SECTION FIRST (before utils import) =====
# Get script and repository paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Add repository root to Python path so we can import src
sys.path.insert(0, REPO_ROOT)

# =========================================================

# NOW import utilities (after path is set)
from src.utils import (
    create_spark_session,
    logger,
    log_section,
    update_log_collection,
    get_logged_files,
)

# =================================================================================================
# Configuration
# =================================================================================================

# Data directory at repository root
LOB_DATA = os.path.join(REPO_ROOT, "lob_data")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "raw"

# Collections
RAW_LOB_COLL = "raw_lob"
INGESTION_LOG_COLL = "ingestion_log"

# Create Spark session with UTC timezone
spark = create_spark_session(
    app_name="Stage1_RawIngestion",
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    driver_memory="4g"
)

# =================================================================================================
# Data Parsing
# =================================================================================================

def parse_lob(df: DataFrame) -> DataFrame:
    """
    Parses LOB data: converts timestamp to naive UTC TimestampType and JSON strings to arrays.
    """
    # Define array structure for bids/asks: array of [price, volume] pairs
    price_volume_pair = ArrayType(ArrayType(DoubleType()))
    
    # Parse arrays
    df = (
        df
        .withColumn("bids", from_json(col("bids").cast("string"), price_volume_pair))
        .withColumn("asks", from_json(col("asks").cast("string"), price_volume_pair))
    )
    
    # Convert timestamp to TimestampType
    # IMPORTANT: Collection script creates timezone-aware timestamps
    # Spark will handle the conversion, but we need UTC timezone set in Spark session
    df = df.withColumn("timestamp", col("timestamp").cast(TimestampType()))
    
    return df

# =================================================================================================
# Main Ingestion Logic
# =================================================================================================

def ingest_raw_lob_data() -> None:
    """
    Ingests raw LOB Parquet files into MongoDB raw collection.
    """
    if not os.path.isdir(LOB_DATA):
        logger(f"LOB data folder not found: {LOB_DATA}. No data to ingest.", level="WARN")
        return
    
    # Get list of already processed files
    already_processed = get_logged_files(spark, DB_NAME, INGESTION_LOG_COLL, "Raw LOB")
    files = sorted([f for f in os.listdir(LOB_DATA) if f.endswith(".parquet")])
    
    if not files:
        logger(f"No Parquet files found in {LOB_DATA}", level="WARN")
        return
    
    logger(f"Found {len(files)} Parquet files, {len(already_processed)} already processed", level="INFO")
    
    total_records = 0
    processed_files = 0

    for file in files:
        if file in already_processed:
            logger(f"Skipping already processed file: {file}", level="INFO")
            continue

        full_path = os.path.join(LOB_DATA, file)
        logger(f"Processing: {file}", level="INFO")

        try:
            # Read and parse LOB data
            df = spark.read.parquet(full_path).dropna()
            df = parse_lob(df)
            
            # Check if file has data
            record_count = df.count()
            if record_count == 0:
                logger(f"Empty file: {file}", level="WARN")
                update_log_collection(spark, DB_NAME, INGESTION_LOG_COLL, file, "Raw LOB", 0)
                continue

            # Upload to MongoDB RAW collection
            # Timestamps will be stored as UTC (Spark session configured for UTC)
            (
                df.write.format("mongodb")
                .option("database", DB_NAME)
                .option("collection", RAW_LOB_COLL)
                .option("replaceDocument", "false")
                .mode("append")
                .save()
            )

            # Update log and counters
            update_log_collection(spark, DB_NAME, INGESTION_LOG_COLL, file, "Raw LOB", record_count)
            total_records += record_count
            processed_files += 1
            
            logger(f"✓ {file} -> {record_count:,} records uploaded to {RAW_LOB_COLL}", level="INFO")
        
        except Exception as e:
            logger(f"Failed to process {file}: {e}", level="ERROR")
            # Continue with next file instead of stopping entire process

    logger(f"STAGE 1 INGESTION SUMMARY: {processed_files} files processed, {total_records:,} total records in {RAW_LOB_COLL}", level="INFO")

# =================================================================================================
# Main Entry Point
# =================================================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    log_section("STAGE 1: Raw LOB Data Ingestion to MongoDB")
    logger(f"Source directory: {os.path.abspath(LOB_DATA)}", level="INFO")
    logger(f"Target collection: {DB_NAME}.{RAW_LOB_COLL}", level="INFO")
    logger(f"Timestamp handling: Collection data (timezone-aware) → MongoDB (naive UTC)", level="INFO")
    log_section("", char="-")
    
    # Stage 1: Ingest raw LOB data
    ingest_raw_lob_data()
    
    end_time = time.time()
    logger(f"STAGE 1 ingestion completed in {end_time - start_time:.2f} seconds.", level="INFO")
    
    spark.stop()