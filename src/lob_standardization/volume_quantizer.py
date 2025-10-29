import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import col, expr, udf
from src.utils.logging import logger


class VolumeQuantizer:
    """Quantizes volumes into logarithmically-spaced bins."""
    
    def __init__(self, spark: SparkSession, B: int, delta: float, epsilon: float = 0.01):
        """
        Initialize volume quantizer.
        
        Args:
            spark: SparkSession instance (needed for broadcast variables)
            B: Number of bins (output will be B+1 bins)
            delta: Price range in standard deviations
            epsilon: Minimum price spacing near zero
        """
        self.spark = spark
        self.B = B
        self.delta = delta
        self.epsilon = epsilon
    
    def quantize_and_aggregate(self, df: DataFrame) -> DataFrame:
        """
        Perform logarithmic quantization and aggregate raw volumes into bins.
        
        Creates B+1 bins ordered by price:
        - Index 0: Most negative price (-delta)
        - Index B//2: Near zero price
        - Index B: Most positive price (+delta)
        
        Args:
            df: DataFrame with 'prices' and 'volumes' columns
            
        Returns:
            DataFrame with 'bins_raw' column containing aggregated volumes
        """
        logger(f'Performing logarithmic quantization with B+1={self.B+1} bins...', "INFO")
        logger(f'Configuration: delta={self.delta}, epsilon={self.epsilon}', "INFO")
        logger(f'Bins ordered by PRICE: index 0 = -{self.delta}, index {self.B//2} approx 0, index {self.B} = +{self.delta}', "INFO")
        logger(f'Aggregating RAW volumes (normalization will come after)', "INFO")
        
        # Broadcast configuration to all workers
        B_broadcast = self.spark.sparkContext.broadcast(self.B)
        delta_broadcast = self.spark.sparkContext.broadcast(self.delta)
        epsilon_broadcast = self.spark.sparkContext.broadcast(self.epsilon)
        
        @udf(ArrayType(DoubleType()))
        def bin_raw_volumes_per_snapshot(prices_arr, volumes_arr):
            """UDF to bin volumes for a single snapshot."""
            B_local = B_broadcast.value
            delta_local = delta_broadcast.value
            epsilon_local = epsilon_broadcast.value
            
            B_total = B_local + 1
            empty_result = [0.0] * B_total
            
            if not prices_arr or not volumes_arr:
                return empty_result
            
            prices = np.array(prices_arr, dtype=np.float64)
            volumes = np.array(volumes_arr, dtype=np.float64)
            
            n = min(len(prices), len(volumes))
            if n == 0:
                return empty_result
            
            prices = prices[:n]
            volumes = volumes[:n]
            
            # Filter valid entries
            valid_mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes > 0)
            prices = prices[valid_mask]
            volumes = volumes[valid_mask]
            
            if len(prices) == 0:
                return empty_result
            
            n_half = B_local // 2
            
            # Create logarithmic spacing from epsilon to delta
            log_points = np.exp(np.linspace(np.log(epsilon_local), np.log(delta_local), n_half))
            
            # Build edges in monotonically increasing order
            negative_edges = -log_points[::-1]  # Reverse to get increasing order
            positive_edges = log_points
            
            edges = np.concatenate([
                [-np.inf],
                negative_edges,
                [0],
                positive_edges,
                [np.inf]
            ])
            
            # Assign prices to bins (lower price → lower index)
            idx = np.digitize(prices, edges, right=False) - 1
            idx = np.clip(idx, 0, B_total - 1)
            
            # Aggregate raw volumes
            bins = np.zeros(B_total, dtype=np.float64)
            np.add.at(bins, idx, volumes)
            
            return bins.tolist()
        
        df = df.withColumn("bins_raw", bin_raw_volumes_per_snapshot(col("prices"), col("volumes")))
        
        logger(f'Created {self.B+1} logarithmically-spaced bins with raw aggregated volumes', "INFO")
        
        return df
    
    @staticmethod
    def normalize_bins(df: DataFrame) -> DataFrame:
        """
        Normalize binned volumes so they sum to 1.0 per snapshot.
        
        Args:
            df: DataFrame with 'bins_raw' column
            
        Returns:
            DataFrame with normalized 'bins' column
        """
        logger('Normalizing binned volumes...', "INFO")
        
        df = df.withColumn(
            "total_volume",
            expr("aggregate(bins_raw, 0D, (acc, x) -> acc + x)")
        )
        
        df = df.withColumn(
            "bins",
            expr("transform(bins_raw, x -> CASE WHEN total_volume = 0 THEN 0D ELSE x / total_volume END)")
        )
        
        logger('Volume normalization complete', "INFO")
        return df.drop("bins_raw", "total_volume")