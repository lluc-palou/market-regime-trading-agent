from src.utils.logging import logger
from pyspark.sql import DataFrame
from pyspark.sql.functions import expr

class DepthFeatures:
    """Calculates depth-based features for various book levels."""
    
    def __init__(self, bands: list):
        """
        Initialize with list of depth bands.
        
        Args:
            bands: List of k values for top-k levels (0 means whole book)
                   Example: [5, 15, 50, 0] for very-near, near, middle, whole
        """
        self.bands = bands
    
    def calculate_all(self, df: DataFrame) -> DataFrame:
        """
        Calculate all depth-based features for all bands.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            
        Returns:
            DataFrame with all depth feature columns added
        """
        result = df
        for k in self.bands:
            logger(f'Calculating depth features for k={k}...', "INFO")
            result = self.calculate_depth_imbalance(result, k)
            result = self.calculate_market_depth(result, k)
            result = self.calculate_liquidity_concentration(result, k)
            result = self.calculate_price_impact_proxy(result, k)
            result = self.calculate_liquidity_spread(result, k)
        return result
    
    @staticmethod
    def calculate_depth_imbalance(df: DataFrame, k: int) -> DataFrame:
        """
        Calculate order book depth imbalance for top-k levels.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            k: Number of levels to consider (0 = whole book)
            
        Returns:
            DataFrame with 'depth_imbalance_*' column added
        """
        K = int(k)
        
        # Convert to struct and sort
        b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
        a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
        b_sorted = f"reverse(array_sort({b_struct}))"
        a_sorted = f"array_sort({a_struct})"
        
        # Select top k levels
        b_topk = f"slice({b_sorted}, 1, {K})" if K > 0 else b_sorted
        a_topk = f"slice({a_sorted}, 1, {K})" if K > 0 else a_sorted
        
        # Sum volumes
        vb = f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y) -> acc + y)"
        va = f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y) -> acc + y)"
        
        # Calculate imbalance
        denom = f"(({vb}) + ({va}))"
        imb = f"CASE WHEN ({denom})=0D THEN 0D ELSE (({vb}) - ({va})) / ({denom}) END"
        
        suffix = f"top_{K}" if K > 0 else "all"
        return df.withColumn(f"depth_imbalance_{suffix}", expr(imb))
    
    @staticmethod
    def calculate_market_depth(df: DataFrame, k: int) -> DataFrame:
        """
        Calculate total market depth for top-k levels.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            k: Number of levels to consider (0 = whole book)
            
        Returns:
            DataFrame with 'depth_*' column added
        """
        K = int(k)
        
        b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
        a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
        b_sorted = f"reverse(array_sort({b_struct}))"
        a_sorted = f"array_sort({a_struct})"
        
        b_topk = f"slice({b_sorted}, 1, {K})" if K > 0 else b_sorted
        a_topk = f"slice({a_sorted}, 1, {K})" if K > 0 else a_sorted
        
        depth_sum = (
            f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
            f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
        )
        
        suffix = f"top_{K}" if K > 0 else "all"
        return df.withColumn(f"depth_{suffix}", expr(depth_sum))
    
    @staticmethod
    def calculate_liquidity_concentration(df: DataFrame, k: int) -> DataFrame:
        """
        Calculate liquidity concentration for top-k levels.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            k: Number of levels to consider (0 = whole book)
            
        Returns:
            DataFrame with 'liquidity_concentration_*' column added
        """
        K = int(k)
        
        b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
        a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
        b_sorted = f"reverse(array_sort({b_struct}))"
        a_sorted = f"array_sort({a_struct})"
        
        b_topk = f"slice({b_sorted}, 1, {K})" if K > 0 else b_sorted
        a_topk = f"slice({a_sorted}, 1, {K})" if K > 0 else a_sorted
        
        # Total volume (whole book)
        denom = (
            f"aggregate(transform({b_sorted}, x -> x.v), 0D, (acc,y)->acc+y) + "
            f"aggregate(transform({a_sorted}, x -> x.v), 0D, (acc,y)->acc+y)"
        )
        
        # Top-k volume
        num = (
            f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
            f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
        )
        
        lc = f"CASE WHEN ({denom})=0D THEN 0D ELSE ({num})/({denom}) END"
        
        suffix = f"top_{K}" if K > 0 else "all"
        return df.withColumn(f"liquidity_concentration_{suffix}", expr(lc))
    
    @staticmethod
    def calculate_price_impact_proxy(df: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
        """
        Calculate price impact proxy for top-k levels.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            k: Number of levels to consider (0 = whole book)
            eps: Small constant to avoid division by zero
            
        Returns:
            DataFrame with 'price_impact_proxy_*' column added
        """
        K = int(k)
        
        b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
        a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
        b_sorted = f"reverse(array_sort({b_struct}))"
        a_sorted = f"array_sort({a_struct})"
        
        # Select price at k-th level
        b_eff = f"CASE WHEN {K} > 0 THEN least({K}, size({b_sorted})) ELSE size({b_sorted}) END"
        a_eff = f"CASE WHEN {K} > 0 THEN least({K}, size({a_sorted})) ELSE size({a_sorted}) END"
        PbK = f"element_at({b_sorted}, {b_eff}).p"
        PaK = f"element_at({a_sorted}, {a_eff}).p"
        
        # Select top-k levels
        b_topk = f"slice({b_sorted}, 1, {b_eff})"
        a_topk = f"slice({a_sorted}, 1, {a_eff})"
        
        # Sum volumes
        depth_sum = (
            f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
            f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
        )
        
        lam = f"CASE WHEN ({depth_sum})<=0D THEN 0D ELSE (({PaK}) - ({PbK})) / ({depth_sum} + {eps}) END"
        
        suffix = f"top_{K}" if K > 0 else "all"
        return df.withColumn(f"price_impact_proxy_{suffix}", expr(lam))
    
    @staticmethod
    def calculate_liquidity_spread(df: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
        """
        Calculate liquidity spread for top-k levels.
        
        Args:
            df: DataFrame with 'bids' and 'asks' columns
            k: Number of levels to consider (0 = whole book)
            eps: Small constant to avoid division by zero
            
        Returns:
            DataFrame with 'liquidity_spread_*' column added
        """
        K = int(k)
        
        b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
        a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
        b_sorted = f"reverse(array_sort({b_struct}))"
        a_sorted = f"array_sort({a_struct})"
        
        # Effective k
        kmin_all = f"least(size({b_sorted}), size({a_sorted}))"
        k_eff = f"CASE WHEN {K} > 0 THEN least({K}, {kmin_all}) ELSE {kmin_all} END"
        b_top = f"slice({b_sorted}, 1, {k_eff})"
        a_top = f"slice({a_sorted}, 1, {k_eff})"
        
        # Extract prices and volumes
        Pa = f"transform({a_top}, x -> x.p)"
        Va = f"transform({a_top}, x -> x.v)"
        Pb = f"transform({b_top}, x -> x.p)"
        Vb = f"transform({b_top}, x -> x.v)"
        
        # Calculate weights and differences
        w = f"zip_with({Va}, {Vb}, (va, vb) -> va + vb)"
        diff = f"zip_with({Pa}, {Pb}, (pa, pb) -> pa - pb)"
        
        # Calculate liquidity spread
        num_arr = f"zip_with({diff}, {w}, (d, ww) -> d * ww)"
        denom = f"aggregate({w}, 0D, (acc, y) -> acc + y)"
        num = f"aggregate({num_arr}, 0D, (acc, y) -> acc + y)"
        lspread = f"CASE WHEN ({denom})<=0D THEN 0D ELSE ({num}) / ({denom} + {eps}) END"
        
        suffix = f"top_{K}" if K > 0 else "all"
        return df.withColumn(f"liquidity_spread_{suffix}", expr(lspread))