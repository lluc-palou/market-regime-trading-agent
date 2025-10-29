from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr
from src.utils.logging import logger


class PriceStandardizer:
    """Standardizes LOB prices relative to mid-price and volatility."""
    
    def __init__(self, eps: float = 1e-8, min_denom: float = 1e-6):
        """
        Initialize price standardizer.
        
        Args:
            eps: Small constant to avoid division by zero
            min_denom: Minimum denominator value
        """
        self.eps = eps
        self.min_denom = min_denom
    
    @staticmethod
    def prepare_ordered_price_volume_pairs(df: DataFrame) -> DataFrame:
        """
        Concatenate bids (reversed) and asks to maintain natural price order.
        
        Creates a single array ordered from negative (far bids) to positive (far asks),
        with mid-price at approximately index B//2.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            
        Returns:
            DataFrame with 'price_volume_pairs' column
        """
        logger('Preparing ordered price-volume pairs...', "INFO")
        
        # Reverse bids to go from low to high price
        # Then concatenate with asks for natural order
        df = df.withColumn("reversed_bids", expr("reverse(bids)"))
        df = df.withColumn("price_volume_pairs", expr("concat(reversed_bids, asks)"))
        df = df.drop("reversed_bids")
        
        logger('Price-volume pairs prepared in natural order', "INFO")
        return df
    
    def standardize_prices(self, df: DataFrame) -> DataFrame:
        """
        Standardize prices in price_volume_pairs array.
        
        Formula: standardized_price = (price - mid_price) / (volatility * mid_price + eps)
        
        Args:
            df: DataFrame with 'price_volume_pairs', 'mid_price', 'volatility' columns
            
        Returns:
            DataFrame with standardized 'price_volume_pairs'
        """
        logger('Standardizing prices...', "INFO")
        
        df = df.withColumn(
            "_safe_denom",
            expr(f"GREATEST(volatility * mid_price, {self.min_denom})")
        )
        
        df = df.withColumn(
            "price_volume_pairs", 
            expr(f"transform(price_volume_pairs, x -> array((x[0] - mid_price) / (_safe_denom + {self.eps}), x[1]))")
        )
        
        logger('Price standardization complete', "INFO")
        return df.drop("_safe_denom")
    
    @staticmethod
    def clip_price_levels(df: DataFrame, delta: float) -> DataFrame:
        """
        Clip price levels to [-delta, +delta] standard deviations from mid-price.
        
        Args:
            df: DataFrame with standardized 'price_volume_pairs'
            delta: Clipping threshold in standard deviations
            
        Returns:
            DataFrame with clipped price levels
        """
        logger(f'Clipping price levels to [±{delta}] std from mid-price...', "INFO")
        
        df = df.withColumn(
            "price_volume_pairs",
            expr(f"filter(price_volume_pairs, x -> x[0] >= -{delta} AND x[0] <= {delta})")
        )
        
        logger('Price level clipping complete', "INFO")
        return df
    
    @staticmethod
    def extract_prices_and_volumes(df: DataFrame) -> DataFrame:
        """
        Extract separate price and volume arrays for quantization.
        
        Args:
            df: DataFrame with 'price_volume_pairs' column
            
        Returns:
            DataFrame with 'prices' and 'volumes' columns
        """
        logger('Extracting prices and volumes arrays...', "INFO")
        
        df = df.withColumn("prices", expr("transform(price_volume_pairs, x -> x[0])"))
        df = df.withColumn("volumes", expr("transform(price_volume_pairs, x -> x[1])"))
        
        logger('Prices and volumes extracted', "INFO")
        return df