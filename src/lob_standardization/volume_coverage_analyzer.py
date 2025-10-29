from pyspark.sql import DataFrame
from pyspark.sql.functions import expr
from src.utils.logging import logger

class CoverageAnalyzer:
    """Analyzes volume coverage for different clipping thresholds."""
    
    @staticmethod
    def analyze_coverage(df: DataFrame, delta: float) -> DataFrame:
        """
        Analyze what percentage of volume falls within the clipping threshold.
        
        Args:
            df: DataFrame with standardized 'price_volume_pairs'
            delta: Clipping threshold to analyze
            
        Returns:
            DataFrame with coverage statistics
        """
        logger(f'Analyzing volume coverage for delta={delta}...', "INFO")
        
        df = df.withColumn(
            "total_volume_before_clip",
            expr("aggregate(price_volume_pairs, 0D, (acc, x) -> acc + x[1])")
        )
        
        df = df.withColumn(
            "clipped_pairs",
            expr(f"filter(price_volume_pairs, x -> x[0] >= -{delta} AND x[0] <= {delta})")
        )
        
        df = df.withColumn(
            "total_volume_after_clip",
            expr("aggregate(clipped_pairs, 0D, (acc, x) -> acc + x[1])")
        )
        
        df = df.withColumn(
            "volume_coverage_pct",
            expr("CASE WHEN total_volume_before_clip = 0 THEN 0D ELSE (total_volume_after_clip / total_volume_before_clip) * 100 END)")
        )
        
        logger('Volume coverage analysis complete', "INFO")
        return df.drop("clipped_pairs")