from datetime import datetime, timedelta
from pyspark.sql import DataFrame, SparkSession

from src.utils.logging import logger
from src.utils.timestamp import format_timestamp_for_mongodb
from src.utils.database import read_from_mongodb

class BatchProcessor:
    """Handles hourly batch processing for split materialization."""
    
    def __init__(self, spark: SparkSession, db_name: str):
        """
        Initialize batch processor.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
        """
        self.spark = spark
        self.db_name = db_name
    
    def get_all_hours(self, input_collection: str) -> list:
        """
        Get all available hours in the input collection.
        
        Args:
            input_collection: Input collection name
            
        Returns:
            List of hour strings in ascending order
        """
        logger('Discovering available hours...', "INFO")
        
        # Load just timestamps to discover hours
        pipeline = [
            {"$project": {"timestamp": 1}},
            {"$addFields": {
                "hour_str": {"$dateToString": {"format": "%Y-%m-%dT%H:00:00.000Z", "date": "$timestamp"}}
            }},
            {"$group": {"_id": "$hour_str"}},
            {"$sort": {"_id": 1}}
        ]
        
        hours_df = read_from_mongodb(self.spark, self.db_name, input_collection, pipeline)
        
        hours = [row._id for row in hours_df.collect()]
        
        if hours:
            logger(f'Found {len(hours)} hours: {hours[0]} to {hours[-1]}', "INFO")
        else:
            logger('No hours found!', "WARNING")
        
        return hours
    
    def load_hour_batch(self, input_collection: str, hour_str: str) -> DataFrame:
        """
        Load a single hour batch from input collection.
        
        Args:
            input_collection: Input collection name
            hour_str: Hour string (e.g., "2024-01-15T10:00:00.000Z")
            
        Returns:
            DataFrame with all documents in that hour
        """
        # Parse hour string
        hour_dt = datetime.fromisoformat(hour_str.replace('Z', ''))
        next_hour_dt = hour_dt + timedelta(hours=1)
        
        pipeline = [
            {"$match": {
                "timestamp": {
                    "$gte": {"$date": format_timestamp_for_mongodb(hour_dt)},
                    "$lt": {"$date": format_timestamp_for_mongodb(next_hour_dt)}
                }
            }},
            {"$sort": {"timestamp": 1}},
            {"$addFields": {
                "timestamp_str": {"$dateToString": {"format": "%Y-%m-%dT%H:%M:%S.%LZ", "date": "$timestamp"}}
            }}
        ]
        
        batch_df = read_from_mongodb(self.spark, self.db_name, input_collection, pipeline)
        
        return batch_df