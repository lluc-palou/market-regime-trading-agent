"""
Null Filtering Script (Stage 11.5)

Removes documents with null values from standardized split collections.
Runs after Stage 11 (feature standardization) and before Stage 12 (VQ-VAE).

Processing Strategy:
- Loads data in hourly batches (same as all other stages)
- Maintains temporal ordering when writing back
- Uses overwrite/append pattern to preserve collection integrity

This ensures the modeling pipeline receives only clean, complete samples
without any null/NaN/Inf values in the features array.

Usage:
    python scripts/11.5_filter_nulls.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import BooleanType

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"  # Read/write to same collections (after Stage 11 cyclic swap)

MAX_SPLITS = 5

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
DRIVER_MEMORY = "4g"

# Temporary collection suffix for safe overwrite
TEMP_SUFFIX = "_output"

# =================================================================================================
# Utility Functions
# =================================================================================================

def get_all_hours(spark: SparkSession, db_name: str, collection: str) -> List[datetime]:
    """
    Get all available hours in a collection.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        collection: Collection name
        
    Returns:
        List of datetime objects representing available hours (sorted)
    """
    logger('Discovering available hours...', "INFO")
    
    pipeline = [
        {"$project": {"timestamp": 1}},
        {"$addFields": {
            "hour_str": {"$dateToString": {"format": "%Y-%m-%dT%H:00:00.000Z", "date": "$timestamp"}}
        }},
        {"$group": {"_id": "$hour_str"}},
        {"$sort": {"_id": 1}}
    ]
    
    hours_df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    hours_list = [datetime.fromisoformat(row._id.replace('Z', '')) for row in hours_df.collect()]
    
    if hours_list:
        logger(f'Found {len(hours_list)} hours: {hours_list[0]} to {hours_list[-1]}', "INFO")
    else:
        logger('No hours found!', "WARNING")
    
    return hours_list


def load_hour_batch(
    spark: SparkSession,
    db_name: str,
    collection: str,
    start_hour: datetime,
    end_hour: datetime
) -> DataFrame:
    """
    Load one hour batch from collection with temporal filtering and ordering.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        collection: Collection name
        start_hour: Start of hour window (inclusive)
        end_hour: End of hour window (exclusive)
        
    Returns:
        DataFrame with documents in the hour window, sorted by timestamp
    """
    start_str = start_hour.isoformat() + 'Z'
    end_str = end_hour.isoformat() + 'Z'
    
    pipeline = [
        {"$match": {
            "timestamp": {
                "$gte": {"$date": start_str},
                "$lt": {"$date": end_str}
            }
        }},
        {"$sort": {"timestamp": 1}}
    ]
    
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    return df


def has_nulls_in_features(features):
    """
    Check if features array contains any null, NaN, or Inf values.
    
    This UDF will be applied to each document's features array to determine
    if it should be filtered out.
    
    Args:
        features: Array of feature values
        
    Returns:
        True if any null/None/NaN/Inf found, False if all values are valid
    """
    if features is None:
        return True
    
    for value in features:
        if value is None:
            return True
        
        try:
            # Check for NaN and Inf
            import math
            if math.isnan(value) or math.isinf(value):
                return True
        except (TypeError, ValueError):
            # If we can't check (shouldn't happen with float), assume corrupted
            return True
    
    return False


# =================================================================================================
# Core Filtering Logic
# =================================================================================================

def filter_split_nulls_hourly(
    spark: SparkSession,
    db_name: str,
    split_id: int,
    collection_prefix: str,
    collection_suffix: str
):
    """
    Filter nulls from a single split collection using hourly batch processing.
    
    This function:
    1. Discovers all hours in the collection
    2. Processes each hour sequentially
    3. Filters documents with null values
    4. Writes clean data to temporary collection
    5. Replaces original collection atomically
    
    Args:
        spark: SparkSession
        db_name: Database name
        split_id: Split ID to process
        collection_prefix: Collection prefix (e.g., 'split_')
        collection_suffix: Collection suffix (e.g., '_input')
    """
    collection_name = f"{collection_prefix}{split_id}{collection_suffix}"
    temp_collection = f"{collection_prefix}{split_id}{TEMP_SUFFIX}"
    
    log_section(f'FILTERING NULLS - SPLIT {split_id}')
    logger(f'Source Collection: {collection_name}', "INFO")
    logger(f'Temp Collection: {temp_collection}', "INFO")
    logger('', "INFO")
    
    # Get all hours in chronological order
    all_hours = get_all_hours(spark, db_name, collection_name)
    
    if not all_hours:
        logger('No data found in collection, skipping', "WARNING")
        return {
            'split_id': split_id,
            'initial_count': 0,
            'final_count': 0,
            'removed_count': 0,
            'removal_pct': 0.0,
            'role_stats': {}
        }
    
    logger(f'Processing {len(all_hours)} hours in chronological order', "INFO")
    logger('', "INFO")
    
    # Register UDF for null detection
    has_nulls_udf = udf(has_nulls_in_features, BooleanType())
    
    # Statistics tracking
    total_processed = 0
    total_filtered = 0
    total_removed = 0
    role_stats = {}
    
    first_batch = True
    
    # Process each hour sequentially
    for hour_idx, start_hour in enumerate(all_hours):
        hour_start_time = time.time()
        end_hour = start_hour + timedelta(hours=1)
        
        # Load hour batch
        hour_df = load_hour_batch(
            spark,
            db_name,
            collection_name,
            start_hour,
            end_hour
        )
        
        hour_count = hour_df.count()
        
        if hour_count == 0:
            if (hour_idx + 1) % 10 == 0 or (hour_idx + 1) == len(all_hours):
                logger(f'  Hour {hour_idx + 1}/{len(all_hours)} '
                       f'({start_hour.strftime("%Y-%m-%d %H:%M")}): '
                       f'Empty batch, skipping', "INFO")
            continue
        
        # Filter documents without nulls
        clean_df = hour_df.filter(~has_nulls_udf(col("features")))
        clean_count = clean_df.count()
        removed_in_hour = hour_count - clean_count
        
        total_processed += hour_count
        total_filtered += clean_count
        total_removed += removed_in_hour
        
        # Track role statistics
        if removed_in_hour > 0:
            hour_roles = hour_df.groupBy("role").count().collect()
            clean_roles = clean_df.groupBy("role").count().collect()
            
            hour_role_dict = {row['role']: row['count'] for row in hour_roles}
            clean_role_dict = {row['role']: row['count'] for row in clean_roles}
            
            for role in hour_role_dict.keys():
                if role not in role_stats:
                    role_stats[role] = {'before': 0, 'after': 0}
                role_stats[role]['before'] += hour_role_dict.get(role, 0)
                role_stats[role]['after'] += clean_role_dict.get(role, 0)
        else:
            # No nulls found, just track totals
            hour_roles = hour_df.groupBy("role").count().collect()
            for row in hour_roles:
                role = row['role']
                count = row['count']
                if role not in role_stats:
                    role_stats[role] = {'before': 0, 'after': 0}
                role_stats[role]['before'] += count
                role_stats[role]['after'] += count
        
        # Write clean data to temporary collection (maintain temporal order)
        if clean_count > 0:
            # Drop _id to avoid conflicts
            if '_id' in clean_df.columns:
                clean_df = clean_df.drop('_id')
            
            # Ensure timestamp ordering
            clean_df = clean_df.orderBy("timestamp")
            
            # Write with append mode after first batch
            write_mode = "overwrite" if first_batch else "append"
            
            (clean_df.write.format("mongodb")
             .option("database", db_name)
             .option("collection", temp_collection)
             .option("ordered", "true")
             .mode(write_mode)
             .save())
            
            first_batch = False
        
        hour_duration = time.time() - hour_start_time
        
        # Log progress every 10 hours or at the end
        if (hour_idx + 1) % 10 == 0 or (hour_idx + 1) == len(all_hours):
            logger(f'  Hour {hour_idx + 1}/{len(all_hours)} '
                   f'({start_hour.strftime("%Y-%m-%d %H:%M")}): '
                   f'{hour_count:,} docs → {clean_count:,} clean '
                   f'({removed_in_hour:,} removed) in {hour_duration:.2f}s',
                   "INFO")
    
    # Calculate statistics
    removal_pct = (total_removed / total_processed * 100) if total_processed > 0 else 0.0
    
    logger('', "INFO")
    logger(f'Split {split_id} Filtering Complete:', "INFO")
    logger(f'  Total processed: {total_processed:,} documents', "INFO")
    logger(f'  Clean documents: {total_filtered:,}', "INFO")
    logger(f'  Removed documents: {total_removed:,} ({removal_pct:.2f}%)', "INFO")
    
    # Log role-specific statistics
    if role_stats:
        logger('', "INFO")
        logger('Documents removed by role:', "INFO")
        for role in sorted(role_stats.keys()):
            before = role_stats[role]['before']
            after = role_stats[role]['after']
            removed = before - after
            pct = (removed / before * 100) if before > 0 else 0.0
            logger(f'  {role}: {removed:,} removed ({pct:.2f}% of {before:,})', 
                   "INFO" if removed > 0 else "INFO")
    
    # Replace original collection with filtered data
    if total_removed > 0:
        logger('', "INFO")
        logger('Replacing original collection with filtered data...', "INFO")
        
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[db_name]
        
        # Drop original collection
        db[collection_name].drop()
        logger(f'  Dropped {collection_name}', "INFO")
        
        # Rename temp collection to original name
        db[temp_collection].rename(collection_name)
        logger(f'  Renamed {temp_collection} → {collection_name}', "INFO")
        
        client.close()
        
        logger('Collection replacement complete', "INFO")
    else:
        logger('', "INFO")
        logger('No nulls found - original collection unchanged', "INFO")
        
        # Clean up temp collection if it exists
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[db_name]
        if temp_collection in db.list_collection_names():
            db[temp_collection].drop()
        client.close()
    
    logger('', "INFO")
    
    return {
        'split_id': split_id,
        'initial_count': total_processed,
        'final_count': total_filtered,
        'removed_count': total_removed,
        'removal_pct': removal_pct,
        'role_stats': role_stats
    }


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    log_section('NULL FILTERING (STAGE 11.5)')
    logger('', "INFO")
    
    logger('This stage removes documents with null/NaN/Inf values from standardized splits', "INFO")
    logger(f'Database: {DB_NAME}', "INFO")
    logger(f'Collections: {COLLECTION_PREFIX}{{0-{MAX_SPLITS-1}}}{COLLECTION_SUFFIX}', "INFO")
    logger(f'Processing {MAX_SPLITS} splits', "INFO")
    logger('', "INFO")
    
    logger('Processing strategy:', "INFO")
    logger('  - Load data in hourly batches', "INFO")
    logger('  - Filter documents with null values in features array', "INFO")
    logger('  - Write to temporary collection with temporal ordering', "INFO")
    logger('  - Atomically replace original collection', "INFO")
    logger('', "INFO")
    
    # Create Spark session
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="Null_Filtering",
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    logger('Spark session created', "INFO")
    logger('', "INFO")
    
    try:
        # Process each split
        split_results = []
        
        for split_id in range(MAX_SPLITS):
            result = filter_split_nulls_hourly(
                spark,
                DB_NAME,
                split_id,
                COLLECTION_PREFIX,
                COLLECTION_SUFFIX
            )
            split_results.append(result)
        
        # =============================================================================
        # Summary Statistics
        # =============================================================================
        
        log_section('NULL FILTERING SUMMARY')
        logger('', "INFO")
        
        total_initial = sum(r['initial_count'] for r in split_results)
        total_final = sum(r['final_count'] for r in split_results)
        total_removed = sum(r['removed_count'] for r in split_results)
        overall_removal_pct = (total_removed / total_initial * 100) if total_initial > 0 else 0.0
        
        logger('Overall Statistics:', "INFO")
        logger(f'  Total documents before: {total_initial:,}', "INFO")
        logger(f'  Total documents after: {total_final:,}', "INFO")
        logger(f'  Total documents removed: {total_removed:,}', "INFO")
        logger(f'  Overall removal rate: {overall_removal_pct:.2f}%', "INFO")
        logger('', "INFO")
        
        logger('Per-Split Statistics:', "INFO")
        for result in split_results:
            logger(f'  Split {result["split_id"]}: '
                   f'{result["removed_count"]:,} removed '
                   f'({result["removal_pct"]:.2f}% of {result["initial_count"]:,})',
                   "INFO")
        logger('', "INFO")
        
        # Aggregate role statistics across all splits
        all_role_stats = {}
        for result in split_results:
            for role, stats in result['role_stats'].items():
                if role not in all_role_stats:
                    all_role_stats[role] = {'before': 0, 'after': 0}
                all_role_stats[role]['before'] += stats['before']
                all_role_stats[role]['after'] += stats['after']
        
        if all_role_stats:
            logger('Aggregate Role Statistics:', "INFO")
            for role in sorted(all_role_stats.keys()):
                before = all_role_stats[role]['before']
                after = all_role_stats[role]['after']
                removed = before - after
                pct = (removed / before * 100) if before > 0 else 0.0
                logger(f'  {role}: {removed:,} removed ({pct:.2f}% of {before:,})',
                       "INFO")
        
        logger('', "INFO")
        log_section('NULL FILTERING COMPLETE')
        logger(f'Next stage (12) will read clean data from {COLLECTION_PREFIX}{{id}}{COLLECTION_SUFFIX}', "INFO")
        logger('All documents are guaranteed to have complete, valid feature arrays', "INFO")
        
    finally:
        spark.stop()
        logger('', "INFO")
        logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    main()