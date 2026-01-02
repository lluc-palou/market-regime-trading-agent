"""
Data Loading Utilities for VQ-VAE Training

Handles loading data from materialized split collections:
- Discover available splits in database
- Generate hourly time windows per split
- Load data in hourly batches with temporal ordering

Pattern follows feature_standardization/data_loader.py
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, min as spark_min, max as spark_max
import torch

from src.utils.logging import logger


def discover_splits(
    spark: SparkSession,
    db_name: str,
    collection_prefix: str = "split_",
    collection_suffix: str = "_input"
) -> List[int]:
    """
    Discover available split collections in database.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        collection_prefix: Prefix for split collections (e.g., 'split_')
        collection_suffix: Suffix for split collections (e.g., '_input')
        
    Returns:
        List of split IDs found in database
    """
    logger('Discovering available splits...', "INFO")
    
    # Get list of collections from MongoDB
    from pymongo import MongoClient
    client = MongoClient(spark.conf.get("spark.mongodb.read.connection.uri"))
    db = client[db_name]
    all_collections = db.list_collection_names()
    client.close()
    
    # Filter to split collections
    split_ids = []
    for coll_name in all_collections:
        if coll_name.startswith(collection_prefix) and coll_name.endswith(collection_suffix):
            # Extract split ID: "split_0_input" -> 0
            try:
                id_part = coll_name[len(collection_prefix):-len(collection_suffix)]
                split_id = int(id_part)
                split_ids.append(split_id)
            except ValueError:
                continue
    
    split_ids.sort()
    
    if split_ids:
        logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")
    else:
        logger('No split collections found!', "WARNING")
    
    return split_ids


def get_split_info(
    spark: SparkSession,
    db_name: str,
    split_collection: str
) -> Dict:
    """
    Get metadata about a split collection.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name (e.g., 'split_0_input')
        
    Returns:
        Dictionary with:
            - min_timestamp: Earliest timestamp
            - max_timestamp: Latest timestamp
            - total_samples: Total number of samples
    """
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", split_collection)
        .load()
    )
    
    # Get time range
    time_stats = df.agg(
        spark_min("timestamp").alias("min_ts"),
        spark_max("timestamp").alias("max_ts")
    ).collect()[0]
    
    # Count samples
    total_samples = df.count()
    
    return {
        'min_timestamp': time_stats['min_ts'],
        'max_timestamp': time_stats['max_ts'],
        'total_samples': total_samples
    }


def get_all_hours(
    spark: SparkSession, 
    db_name: str, 
    split_collection: str
) -> List[datetime]:
    """
    Get all available hours in a split collection.
    
    Uses MongoDB aggregation to efficiently discover hour boundaries.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name (e.g., 'split_0_input')
        
    Returns:
        List of datetime objects representing hour starts (sorted)
    """
    logger(f'Discovering available hours in {split_collection}...', "INFO")
    
    # MongoDB aggregation pipeline to group by hour
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
        .option("collection", split_collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    hours_list = [datetime.fromisoformat(row._id.replace('Z', '')) for row in hours_df.collect()]
    
    if hours_list:
        logger(f'Found {len(hours_list)} hours: {hours_list[0]} to {hours_list[-1]}', "INFO")
    else:
        logger(f'No hours found in {split_collection}!', "WARNING")
    
    return hours_list


def load_hourly_batch(
    spark: SparkSession,
    db_name: str,
    split_collection: str,
    hour_start: datetime,
    hour_end: datetime,
    role: str = None
) -> Optional[torch.Tensor]:
    """
    Load one hour batch from split collection with TEMPORAL ORDERING.

    Critical: Uses MongoDB aggregation with $sort to ensure temporal sequence.

    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name
        hour_start: Start of hour window
        hour_end: End of hour window
        role: Filter by role ('train', 'train_warmup', or 'validation'), or None for all roles

    Returns:
        torch.Tensor of LOB vectors (batch_size, B) or None if empty
    """
    import time
    load_start = time.time()

    start_str = hour_start.isoformat() + 'Z'
    end_str = hour_end.isoformat() + 'Z'

    # Build match filter
    match_filter = {
        "timestamp": {
            "$gte": {"$date": start_str},
            "$lt": {"$date": end_str}
        }
    }

    # Only filter by role if specified
    if role is not None:
        match_filter["role"] = role

    # MongoDB aggregation pipeline with temporal ordering
    pipeline = [
        {"$match": match_filter},
        {"$sort": {"timestamp": 1}},  # CRITICAL: Temporal ordering
        {"$project": {"bins": 1}}
    ]

    query_start = time.time()
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", split_collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )

    count_start = time.time()
    count = df.count()
    count_time = time.time() - count_start

    if count == 0:
        return None

    # Collect LOB vectors
    collect_start = time.time()
    rows = df.collect()
    collect_time = time.time() - collect_start

    bins = [row['bins'] for row in rows]

    # Convert to tensor
    tensor = torch.tensor(bins, dtype=torch.float32)

    total_time = time.time() - load_start

    # Log timing for first few batches to diagnose bottleneck
    if total_time > 2.0:  # Only log slow queries
        logger(f'  [SLOW] Loaded {count} samples in {total_time:.2f}s (count={count_time:.2f}s, collect={collect_time:.2f}s)', "WARNING")

    return tensor


def load_hourly_batch_dataframe(
    spark: SparkSession,
    db_name: str,
    split_collection: str,
    hour_start: datetime,
    hour_end: datetime,
    role: str
) -> DataFrame:
    """
    Load one hour batch as DataFrame (for validation metrics tracking).
    
    Similar to load_hourly_batch but returns DataFrame instead of tensor.
    Useful when we need to track additional metadata.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name
        hour_start: Start of hour window
        hour_end: End of hour window
        role: Filter by role ('train', 'train_warmup', or 'validation')
        
    Returns:
        DataFrame with LOB vectors and metadata
    """
    start_str = hour_start.isoformat() + 'Z'
    end_str = hour_end.isoformat() + 'Z'
    
    pipeline = [
        {"$match": {
            "timestamp": {
                "$gte": {"$date": start_str},
                "$lt": {"$date": end_str}
            },
            "role": role
        }},
        {"$sort": {"timestamp": 1}}
    ]
    
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", split_collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    return df