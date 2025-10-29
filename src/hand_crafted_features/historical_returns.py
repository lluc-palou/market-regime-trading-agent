from src.utils.logging import logger
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, log, lag

class HistoricalReturns:
    """Calculates historical log returns at various lags."""
    
    def __init__(self, lags: list):
        """
        Initialize with list of lag periods.
        
        Args:
            lags: List of lag periods (e.g., [1, 2, 3, 5, 10, 20])
        """
        self.lags = lags
    
    def calculate(self, df: DataFrame) -> DataFrame:
        """
        Calculate historical log returns for all specified lags.
        
        Args:
            df: DataFrame with 'mid_price' and 'timestamp_str' columns
            
        Returns:
            DataFrame with 'past_logret_N' columns added for each lag N
        """
        result = df
        for lag_n in self.lags:
            logger(f'Calculating past_logret_{lag_n}...', "INFO")
            result = self._calculate_single_lag(result, lag_n)
        return result
    
    @staticmethod
    def _calculate_single_lag(df: DataFrame, n: int) -> DataFrame:
        """
        Calculate historical log return for a single lag.
        
        Args:
            df: DataFrame with 'mid_price' and 'timestamp_str' columns
            n: Lag period
            
        Returns:
            DataFrame with 'past_logret_{n}' column added
        """
        w = Window.orderBy("timestamp_str")
        past = lag(col("mid_price"), n).over(w)
        return df.withColumn(f"past_logret_{n}", log(col("mid_price")) - log(past))