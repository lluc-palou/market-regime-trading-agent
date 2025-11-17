import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, expr, lit, array
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

        # Precompute bin edges once (same for all snapshots)
        self._compute_bin_edges()

    def _compute_bin_edges(self):
        """Precompute logarithmic bin edges (same for all snapshots)."""
        n_half = self.B // 2

        # Create logarithmic spacing from epsilon to delta
        log_points = np.exp(np.linspace(np.log(self.epsilon), np.log(self.delta), n_half))

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

        self.edges = edges.tolist()
        logger(f'Precomputed {len(self.edges)} bin edges', "INFO")

    def quantize_and_aggregate(self, df: DataFrame) -> DataFrame:
        """
        Perform logarithmic quantization and aggregate raw volumes into bins using native Spark SQL.

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
        logger('Using native Spark SQL (no UDFs) for better performance on large datasets', "INFO")

        B_total = self.B + 1

        # Create bin assignment function using CASE WHEN based on precomputed edges
        # Each bin is defined by edges[i] <= price < edges[i+1]
        bin_assignment_cases = []
        for bin_idx in range(B_total):
            lower = self.edges[bin_idx]
            upper = self.edges[bin_idx + 1]

            # Handle infinity cases
            if np.isinf(lower) and lower < 0:
                condition = f"price < {upper}"
            elif np.isinf(upper):
                condition = f"price >= {lower}"
            else:
                condition = f"(price >= {lower} AND price < {upper})"

            bin_assignment_cases.append(f"WHEN {condition} THEN {bin_idx}")

        bin_assignment_expr = "CASE " + " ".join(bin_assignment_cases) + f" ELSE {B_total - 1} END"

        # Step 1: Zip prices and volumes into pairs, filter valid ones, assign bins
        df = df.withColumn(
            "price_volume_bin_pairs",
            expr(f"""
                transform(
                    filter(
                        arrays_zip(prices, volumes),
                        pair -> pair.prices IS NOT NULL
                            AND pair.volumes IS NOT NULL
                            AND NOT isnan(pair.prices)
                            AND NOT isnan(pair.volumes)
                            AND NOT isinf(pair.prices)
                            AND NOT isinf(pair.volumes)
                            AND pair.volumes > 0
                    ),
                    pair -> named_struct(
                        'bin_idx',
                        (
                            {bin_assignment_expr.replace('price', 'pair.prices')}
                        ),
                        'volume',
                        pair.volumes
                    )
                )
            """)
        )

        # Step 2: Aggregate volumes by bin index
        # Create an array of B_total bins, initialized to 0
        df = df.withColumn(
            "bins_raw",
            expr(f"""
                aggregate(
                    price_volume_bin_pairs,
                    array_repeat(0.0, {B_total}),
                    (acc, pair) -> transform(
                        sequence(0, {B_total - 1}),
                        idx -> CASE
                            WHEN idx = pair.bin_idx THEN element_at(acc, idx + 1) + pair.volume
                            ELSE element_at(acc, idx + 1)
                        END
                    )
                )
            """)
        )

        # Drop intermediate column
        df = df.drop("price_volume_bin_pairs")

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