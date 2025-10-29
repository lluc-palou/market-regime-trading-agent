"""
Data Loading for Feature Transformation Selection

Handles loading data from split collections for transformation selection.
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
        split_collection: Split collection name (e.g., 'split_0')
        
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


def load_hour_batch(spark: SparkSession, db_name: str, split_collection: str, 
                    start_hour: datetime, end_hour: datetime) -> DataFrame:
    """
    Load one hour batch from split collection.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name
        start_hour: Start of hour window
        end_hour: End of hour window
        
    Returns:
        DataFrame with documents in the hour window
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
    first_row = df.select("feature_names").first()
    
    if first_row is None:
        raise ValueError("No data found in collection")
    
    feature_names = first_row['feature_names']
    
    logger(f'Identified {len(feature_names)} features from feature_names array', "INFO")
    
    return feature_names