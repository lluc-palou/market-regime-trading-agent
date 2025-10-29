from datetime import timedelta
from pyspark.sql import DataFrame
from src.utils.logging import logger
from pyspark.sql.functions import col, expr, lit
from src.utils.timestamp import parse_hour_string, add_hours, format_timestamp_for_mongodb

class BatchLoader:
    """Handles loading of hourly batches with required context windows."""
    
    def __init__(self, required_past_hours: int, required_future_hours: int):
        """
        Initialize batch loader with context window requirements.
        
        Args:
            required_past_hours: Hours of past context needed
            required_future_hours: Hours of future context needed
        """
        self.required_past_hours = required_past_hours
        self.required_future_hours = required_future_hours
    
    def get_processable_hours(self, df: DataFrame) -> list:
        """
        Determine which hours can be fully processed.
        Excludes hours where we don't have enough past or future context.
        
        Args:
            df: DataFrame with 'timestamp_str' column
            
        Returns:
            List of hour strings that can be fully processed
        """
        all_hours = (
            df.withColumn("trading_hour_str", expr("substring(timestamp_str, 1, 13) || ':00:00.000Z'"))
            .select("trading_hour_str")
            .distinct()
            .orderBy("trading_hour_str")
            .collect()
        )
        
        available_hours = [row.trading_hour_str for row in all_hours]
        
        logger(f'Available hours: {len(available_hours)} from {available_hours[0]} to {available_hours[-1]}', "INFO")
        
        processable_hours = []
        
        for i, hour in enumerate(available_hours):
            past_hours_available = i
            future_hours_available = len(available_hours) - i - 1
            
            if (past_hours_available >= self.required_past_hours and 
                future_hours_available >= self.required_future_hours):
                processable_hours.append(hour)
        
        logger(f'Processable hours: {len(processable_hours)}', "INFO")
        if processable_hours:
            logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}', "INFO")
        
        return processable_hours
    
    def load_batch(self, df: DataFrame, target_hour_str: str) -> DataFrame:
        """
        Load batch with required past and future context.
        
        Args:
            df: Full DataFrame with 'timestamp_str' column
            target_hour_str: Target hour to process
            
        Returns:
            DataFrame containing target hour plus required context
        """
        # Parse target hour
        target_dt = parse_hour_string(target_hour_str)
        
        # Calculate boundaries
        past_start_dt = add_hours(target_dt, -self.required_past_hours)
        future_end_dt = add_hours(target_dt, self.required_future_hours + 1)
        
        # Format as strings for filtering
        past_start_str = format_timestamp_for_mongodb(past_start_dt)
        future_end_str = format_timestamp_for_mongodb(future_end_dt)
        
        filter_condition = (
            (col("timestamp_str") >= lit(past_start_str)) &
            (col("timestamp_str") < lit(future_end_str))
        )
        
        logger(f'Loading batch for {target_hour_str}', "INFO")
        logger(f'Window: {past_start_str} to {future_end_str}', "INFO")
        
        return df.filter(filter_condition)