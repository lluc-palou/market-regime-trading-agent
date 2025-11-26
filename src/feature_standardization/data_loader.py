"""
Data Loading for EWMA Half-Life Selection

Handles loading data from split collections for half-life selection.
Reuses patterns from feature transformation module.
"""

from datetime import datetime, timedelta
from typing import List
from pyspark.sql import SparkSession, DataFrame

from src.utils.logging import logger


def get_all_hours(spark: SparkSession, db_name: str, split_collection: str) -> List[datetime]:
    """
    Get all available hours in a split collection.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name (e.g., 'split_0_output')
        
    Returns:
        List of datetime objects representing available hours
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
        .option("collection", split_collection)
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
    split_collection: str,
    start_hour: datetime,
    end_hour: datetime
) -> DataFrame:
    """
    Load one hour batch from split collection.

    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name
        start_hour: Start of hour window
        end_hour: End of hour window

    Returns:
        DataFrame with documents in the hour window (deduplicated by timestamp)
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
        .option("collection", split_collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )

    return df


def identify_feature_names(df: DataFrame) -> List[str]:
    """
    Get feature names from the feature_names array in the first document.
    
    Args:
        df: DataFrame from split collection
        
    Returns:
        List of feature names
        
    Raises:
        ValueError: If no data found in collection
    """
    # Load first document (don't select specific columns yet)
    first_row = df.first()
    
    if first_row is None:
        raise ValueError("No data found in collection")
    
    # Check if feature_names field exists
    if 'feature_names' not in first_row.asDict():
        raise ValueError("feature_names field not found in document")
    
    feature_names = first_row['feature_names']
    
    logger(f'Identified {len(feature_names)} features from feature_names array', "INFO")
    
    return feature_names


def identify_feature_names_from_collection(
    spark: SparkSession,
    db_name: str,
    collection: str
) -> List[str]:
    """
    Load feature names directly from collection using aggregation pipeline.

    This is MORE RELIABLE than identify_feature_names() because it uses
    MongoDB aggregation to explicitly project the feature_names field,
    avoiding Spark schema inference issues with array fields.

    USE THIS FUNCTION if identify_feature_names() fails with collection not found
    or feature_names field not found errors.

    Args:
        spark: SparkSession instance
        db_name: Database name
        collection: Collection name

    Returns:
        List of feature names
    """
    logger(f'Loading feature_names from {collection} using aggregation...', "INFO")

    # Use aggregation pipeline to explicitly project feature_names
    pipeline = [
        {"$limit": 1},
        {"$project": {
            "feature_names": 1,
            "_id": 0
        }}
    ]

    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )

    if df.count() == 0:
        raise ValueError(f"Collection {collection} is empty or does not exist")

    first_row = df.first()

    if first_row is None:
        raise ValueError(f"Could not retrieve first document from {collection}")

    row_dict = first_row.asDict()

    if 'feature_names' not in row_dict:
        raise ValueError(
            f"feature_names field not found in collection {collection}.\n"
            f"Available fields: {list(row_dict.keys())}\n"
            f"Make sure Stage 8 (transformation application) completed successfully."
        )

    feature_names = row_dict['feature_names']

    if feature_names is None or len(feature_names) == 0:
        raise ValueError(f"feature_names is empty in collection {collection}")

    logger(f'Identified {len(feature_names)} features', "INFO")

    return list(feature_names)


def filter_standardizable_features(feature_names: List[str]) -> List[str]:
    """
    Filter feature names to only include those that should be standardized.
    
    Excludes features that:
    - Are targets (forward returns)
    - Need to keep original scale (volatility, variance_proxy)
    - Will be dropped before materialization (mid_price, log_return, spread)
    - Raw LOB data (bins)
    
    Args:
        feature_names: List of all feature names
        
    Returns:
        List of feature names to standardize
    """
    EXCLUDE_PATTERNS = [
        'fwd_logret_',      # Forward returns (targets)
    ]
    
    EXCLUDE_EXACT = [
        'bins',             # Binned LOB data (already standardized)
        'variance_proxy',   # Keep original scale
        'volatility',       # Keep original scale
        'mid_price',        # Will be dropped - skip processing
        'log_return',       # Will be dropped - skip processing
        'spread',           # Will be dropped - skip processing
    ]
    
    standardizable = []
    
    for feat_name in feature_names:
        # Skip exact matches
        if feat_name in EXCLUDE_EXACT:
            continue
        
        # Skip pattern matches
        if any(feat_name.startswith(pattern) for pattern in EXCLUDE_PATTERNS):
            continue
        
        standardizable.append(feat_name)
    
    excluded_count = len(feature_names) - len(standardizable)
    logger(f'Filtered to {len(standardizable)} standardizable features (excluded {excluded_count})', "INFO")
    
    if excluded_count > 0:
        logger(f'Excluded features: targets, volatility features, intermediate features', "INFO")
    
    return standardizable