from pyspark.sql import DataFrame
from src.utils.logging import logger
from pyspark.sql.window import Window
from pyspark.sql.functions import col, log, lead

class ForwardReturnsCalculator:
    """Calculates forward log-returns (targets) at various horizons."""
    
    def __init__(self, horizons: list, decision_lag: int = 1):
        """
        Initialize with list of forecast horizons and decision lag.
        
        Args:
            horizons: List of forecast horizons (e.g., [2, 3, 5, 10, 20])
            decision_lag: Lag for decision making (default: 1)
        """
        self.horizons = horizons
        self.decision_lag = decision_lag
    
    def calculate(self, df: DataFrame) -> DataFrame:
        """
        Calculate forward returns for all specified horizons.
        
        Args:
            df: DataFrame with 'mid_price' and 'timestamp_str' columns
            
        Returns:
            DataFrame with 'fwd_logret_N' columns added for each horizon N
        """
        result = df
        for horizon in self.horizons:
            logger(f'Calculating fwd_logret_{horizon}...', "INFO")
            result = self._calculate_single_horizon(result, horizon)
        return result
    
    def _calculate_single_horizon(self, df: DataFrame, n: int) -> DataFrame:
        """
        Calculate forward log-return for a single horizon.
        
        Args:
            df: DataFrame with 'mid_price' and 'timestamp_str' columns
            n: Forecast horizon
            
        Returns:
            DataFrame with 'fwd_logret_{n}' column added
        """
        w = Window.orderBy("timestamp_str")
        base = lead(col("mid_price"), self.decision_lag).over(w)
        future = lead(col("mid_price"), self.decision_lag + n).over(w)
        return df.withColumn(f"fwd_logret_{n}", log(future) - log(base))