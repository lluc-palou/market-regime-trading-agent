from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, expr, log, lag, when

class PriceFeatures:
    """Calculates price-based features from LOB data."""
    
    @staticmethod
    def calculate_mid_prices(df: DataFrame) -> DataFrame:
        """
        Calculate mid-price from LOB data.
        Uses aggregate with greatest/least to find best bid/ask.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            
        Returns:
            DataFrame with 'mid_price' column added
        """
        best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
        best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
        return df.withColumn("mid_price", (best_bid + best_ask) / 2.0)
    
    @staticmethod
    def calculate_log_returns(df: DataFrame) -> DataFrame:
        """
        Calculates log returns from mid-price.
        Works with timestamp_str for ordering.
        
        Args:
            df: DataFrame with 'mid_price' and 'timestamp_str' columns
            
        Returns:
            DataFrame with 'log_return' column added
        """
        w = Window.orderBy("timestamp_str")
        return df.withColumn(
            "log_return", 
            log(col("mid_price")) - log(lag(col("mid_price"), 1).over(w))
        )
    
    @staticmethod
    def calculate_spread(df: DataFrame) -> DataFrame:
        """
        Calculates bid-ask spread.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            
        Returns:
            DataFrame with 'spread' column added
        """
        best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
        best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
        return df.withColumn("spread", best_ask - best_bid)
    
    @staticmethod
    def calculate_microprice(df: DataFrame) -> DataFrame:
        """
        Calculates volume-weighted microprice.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            
        Returns:
            DataFrame with 'microprice' column added
        """
        # Find best prices
        best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
        best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
        df = df.withColumn("best_bid", best_bid).withColumn("best_ask", best_ask)
        
        # Sum volumes at best prices
        best_bid_vol = expr("aggregate(filter(bids, x -> x[0] = best_bid), 0D, (acc,x) -> acc + x[1])")
        best_ask_vol = expr("aggregate(filter(asks, x -> x[0] = best_ask), 0D, (acc,x) -> acc + x[1])")
        df = df.withColumn("best_bid_vol", best_bid_vol).withColumn("best_ask_vol", best_ask_vol)
        
        # Calculate microprice
        micro_expr = when(
            (col("best_bid_vol") + col("best_ask_vol")) == 0.0,
            (col("best_bid") + col("best_ask")) / 2.0
        ).otherwise(
            (col("best_ask") * col("best_bid_vol") + col("best_bid") * col("best_ask_vol")) /
            (col("best_bid_vol") + col("best_ask_vol"))
        )
        
        df = df.withColumn("microprice", micro_expr)
        return df.drop("best_bid", "best_ask", "best_bid_vol", "best_ask_vol")