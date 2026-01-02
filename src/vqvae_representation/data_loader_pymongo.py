"""
Fast Data Loading Using Direct PyMongo (Bypass Spark)

Replaces Spark-MongoDB connector with direct PyMongo queries for 10-50× speedup.
Spark is overkill for single-hour queries - PyMongo is much faster.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import torch
from pymongo import MongoClient

from src.utils.logging import logger


def get_all_hours_pymongo(
    mongo_uri: str,
    db_name: str,
    split_collection: str
) -> List[datetime]:
    """
    Get all available hours using direct MongoDB aggregation (fast).

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Database name
        split_collection: Split collection name

    Returns:
        List of datetime objects representing hour starts (sorted)
    """
    logger(f'Discovering available hours in {split_collection}...', "INFO")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[split_collection]

    # MongoDB aggregation pipeline to group by hour
    pipeline = [
        {"$project": {"timestamp": 1}},
        {"$addFields": {
            "hour_str": {"$dateToString": {"format": "%Y-%m-%dT%H:00:00.000Z", "date": "$timestamp"}}
        }},
        {"$group": {"_id": "$hour_str"}},
        {"$sort": {"_id": 1}}
    ]

    result = list(collection.aggregate(pipeline, allowDiskUse=True))
    hours_list = [datetime.fromisoformat(doc['_id'].replace('Z', '')) for doc in result]

    client.close()

    if hours_list:
        logger(f'Found {len(hours_list)} hours: {hours_list[0]} to {hours_list[-1]}', "INFO")
    else:
        logger(f'No hours found in {split_collection}!', "WARNING")

    return hours_list


def load_hourly_batch_pymongo(
    mongo_uri: str,
    db_name: str,
    split_collection: str,
    hour_start: datetime,
    hour_end: datetime,
    role: str
) -> Optional[torch.Tensor]:
    """
    Load one hour batch using direct PyMongo (10-50× faster than Spark).

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Database name
        split_collection: Split collection name
        hour_start: Start of hour window
        hour_end: End of hour window
        role: Filter by role ('train', 'train_warmup', or 'validation')

    Returns:
        torch.Tensor of LOB vectors (batch_size, B) or None if empty
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[split_collection]

    # Query with timestamp filter and sort
    query = {
        "timestamp": {
            "$gte": hour_start,
            "$lt": hour_end
        },
        "role": role
    }

    # Project only bins field, sort by timestamp
    cursor = collection.find(
        query,
        {"bins": 1, "_id": 0}
    ).sort("timestamp", 1)

    # Collect bins
    docs = list(cursor)
    client.close()

    if not docs:
        return None

    # Extract bins and convert to tensor
    bins = [doc['bins'] for doc in docs]
    tensor = torch.tensor(bins, dtype=torch.float32)

    return tensor


def load_multiple_hours_pymongo(
    mongo_uri: str,
    db_name: str,
    split_collection: str,
    hours: List[datetime],
    role: str = None
) -> Optional[torch.Tensor]:
    """
    Load multiple hours at once using $in query (even faster).

    This is more efficient than loading hours individually because it makes
    a single MongoDB query instead of N queries.

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Database name
        split_collection: Split collection name
        hours: List of hour start times to load
        role: Filter by role (default: None = load all roles)

    Returns:
        torch.Tensor of LOB vectors or None if empty
    """
    if not hours:
        return None

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[split_collection]

    # Build list of time ranges for $or query
    time_ranges = []
    for hour_start in hours:
        hour_end = hour_start + timedelta(hours=1)
        time_ranges.append({
            "timestamp": {
                "$gte": hour_start,
                "$lt": hour_end
            }
        })

    # Single query for all hours
    query = {
        "$or": time_ranges
    }

    # Only filter by role if specified
    if role is not None:
        query["role"] = role

    # Project only bins, sort by timestamp
    cursor = collection.find(
        query,
        {"bins": 1, "_id": 0}
    ).sort("timestamp", 1)

    # Collect all documents
    docs = list(cursor)
    client.close()

    if not docs:
        return None

    # Extract bins and convert to tensor
    bins = [doc['bins'] for doc in docs]
    tensor = torch.tensor(bins, dtype=torch.float32)

    return tensor
