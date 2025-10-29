from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, expr, pow, row_number, lit, when, sum as spark_sum

class VolatilityFeatures:
    """Calculates variance and volatility using EWMA approach."""
    
    def __init__(self, half_life: int = 20):
        """
        Initialize with EWMA half-life parameter.
        
        Args:
            half_life: Half-life for EWMA calculation (default: 20)
        """
        self.half_life = half_life
    
    def calculate(self, df: DataFrame) -> DataFrame:
        """
        Calculate variance estimator and volatility.
        
        Args:
            df: DataFrame with 'log_return' and 'timestamp_str' columns
            
        Returns:
            DataFrame with 'variance_proxy' and 'volatility' columns added
        """
        df = self._estimate_variance(df)
        df = self._calculate_volatility(df)
        return df
    
    def _estimate_variance(self, df: DataFrame) -> DataFrame:
        """
        Calculate variance estimator using EWMA.
        
        Args:
            df: DataFrame with 'log_return' and 'timestamp_str' columns
            
        Returns:
            DataFrame with 'variance_proxy' column added
        """
        alpha = 1.0 - pow(lit(2.0), lit(-1.0) / lit(float(self.half_life)))
        beta = (lit(1.0) - alpha)
        w = Window.orderBy("timestamp_str")
        rn = row_number().over(w) - lit(1)
        
        df = df.withColumn("r2", col("log_return") * col("log_return"))
        df = df.withColumn("z_i", when(col("r2").isNotNull(), col("r2") * pow(beta, -rn)).otherwise(lit(None)))
        
        wcum = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        df = df.withColumn("cum_z", spark_sum("z_i").over(wcum))
        
        df = df.withColumn("variance_proxy", alpha * pow(beta, rn) * col("cum_z"))
        df = df.drop("r2", "z_i", "cum_z")
        
        return df
    
    @staticmethod
    def _calculate_volatility(df: DataFrame) -> DataFrame:
        """
        Calculate volatility from variance estimator.
        
        Args:
            df: DataFrame with 'variance_proxy' column
            
        Returns:
            DataFrame with 'volatility' column added
        """
        return df.withColumn("volatility", expr("sqrt(variance_proxy)"))