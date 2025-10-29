from datetime import timedelta
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, lit, when

from src.utils.logging import logger
from src.utils.timestamp import parse_hour_string, add_hours
from src.hand_crafted_features.price_features import PriceFeatures
from src.hand_crafted_features.volatility_features import VolatilityFeatures

class BatchProcessor:
    """Processes hourly batches for LOB standardization."""
    
    def __init__(self, required_past_hours: int):
        """
        Initialize batch processor.
        
        Args:
            required_past_hours: Hours of past context needed for volatility calculation
        """
        self.required_past_hours = required_past_hours
        self.price_features = PriceFeatures()
        self.volatility_features = VolatilityFeatures(half_life=20)
    
    def get_processable_hours(self, df: DataFrame) -> list:
        """
        Determine which hours can be processed.
        
        Args:
            df: DataFrame with 'timestamp_str' column
            
        Returns:
            List of processable hour strings
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
            
            if past_hours_available >= self.required_past_hours:
                processable_hours.append(hour)
        
        logger(f'Processable hours: {len(processable_hours)}', "INFO")
        if processable_hours:
            logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}', "INFO")
        
        return processable_hours
    
    def load_batch(self, df: DataFrame, target_hour_str: str) -> DataFrame:
        """
        Load batch with required past context.
        
        Args:
            df: Full DataFrame
            target_hour_str: Target hour to process
            
        Returns:
            DataFrame with batch data
        """
        target_dt = parse_hour_string(target_hour_str)
        past_start_dt = add_hours(target_dt, -self.required_past_hours)
        future_end_dt = add_hours(target_dt, 1)
        
        past_start_str = past_start_dt.isoformat() + '.000Z'
        future_end_str = future_end_dt.isoformat() + '.000Z'
        
        filter_condition = (
            (col("timestamp_str") >= lit(past_start_str)) &
            (col("timestamp_str") < lit(future_end_str))
        )
        
        logger(f'Loading batch for {target_hour_str}', "INFO")
        logger(f'Window: {past_start_str} to {future_end_str}', "INFO")
        
        return df.filter(filter_condition)
    
    def calculate_price_features_for_batch(self, df: DataFrame) -> DataFrame:
        """
        Calculate mid-price and volatility for standardization.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            
        Returns:
            DataFrame with 'mid_price' and 'volatility' columns
        """
        logger('Calculating price features (mid_price, log_return, volatility)...', "INFO")
        
        df = self.price_features.calculate_mid_prices(df)
        df = self.price_features.calculate_log_returns(df)
        df = self.volatility_features.calculate(df)
        
        return df.select("timestamp_str", "mid_price", "log_return", "variance_proxy", "volatility")
    
    def filter_to_target_hour(self, df: DataFrame, target_hour_str: str, is_first_batch: bool) -> DataFrame:
        """
        Filter dataframe to appropriate time window.
        
        Args:
            df: DataFrame to filter
            target_hour_str: Target hour string
            is_first_batch: Whether this is the first batch
            
        Returns:
            Filtered DataFrame
        """
        df = df.withColumn(
            "is_context", 
            when(expr("substring(timestamp_str, 1, 13) || ':00:00.000Z'") < lit(target_hour_str), lit(True)).otherwise(lit(False))
        )
        
        if is_first_batch:
            target_dt = parse_hour_string(target_hour_str)
            context_end_dt = add_hours(target_dt, 1)
            context_end_str = context_end_dt.isoformat() + '.000Z'
            
            df_output = df.filter(col("timestamp_str") < lit(context_end_str))
            context_cnt = df_output.filter(col("is_context") == True).count()
            target_cnt = df_output.filter(col("is_context") == False).count()
            logger(f'First batch - keeping context: {context_cnt} context rows, {target_cnt} target rows', "INFO")
        else:
            df_output = df.filter(expr("substring(timestamp_str, 1, 13) || ':00:00.000Z'") == lit(target_hour_str))
            target_cnt = df_output.count()
            logger(f'Regular batch - keeping target hour only: {target_cnt} rows', "INFO")
        
        return df_output.drop("is_context")