import time
from datetime import timedelta
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, expr, lit

from src.utils.logging import logger
from src.utils.database import read_sorted_with_timestamp_strings, write_with_timestamp_conversion
from src.utils.timestamp import parse_hour_string, add_hours

from .price_features import PriceFeatures
from .historical_returns import HistoricalReturns
from .forward_returns import ForwardReturnsCalculator
from .volatility_features import VolatilityFeatures
from .depth_features import DepthFeatures
from .batch_loader import BatchLoader

class FeatureOrchestrator:
    """Orchestrates the entire feature engineering pipeline."""
    
    def __init__(self, spark: SparkSession, db_name: str, input_collection: str, 
                 output_collection: str, config: dict):
        """
        Initialize feature orchestrator.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            input_collection: Input collection name
            output_collection: Output collection name
            config: Configuration dictionary with feature parameters
        """
        self.spark = spark
        self.db_name = db_name
        self.input_collection = input_collection
        self.output_collection = output_collection
        self.config = config
        
        # Initialize components
        self.batch_loader = BatchLoader(
            config['required_past_hours'],
            config['required_future_hours']
        )
        self.price_features = PriceFeatures()
        self.volatility_features = VolatilityFeatures(config['variance_half_life'])
        self.depth_features = DepthFeatures(config['depth_bands'])
        self.forward_returns = ForwardReturnsCalculator(
            config['forward_horizons'],
            config['decision_lag']
        )
        self.historical_returns = HistoricalReturns(config['historical_lags'])
        
        self.raw_lob = None
        self.processable_hours = None
    
    def load_raw_data(self):
        """Load raw LOB data from MongoDB."""
        logger(f'Loading raw LOB data from {self.input_collection}', "INFO")
        
        additional_fields = ["bids", "asks", "fold_type", "fold_id", "split_roles"]
        
        self.raw_lob = read_sorted_with_timestamp_strings(
            self.spark,
            self.db_name,
            self.input_collection,
            additional_fields
        )
        
        count = self.raw_lob.count()
        logger(f'Loaded {count} raw LOB records', "INFO")
    
    def determine_processable_hours(self):
        """Determine which hours can be processed."""
        self.processable_hours = self.batch_loader.get_processable_hours(self.raw_lob)
        
        if not self.processable_hours:
            raise ValueError('No processable hours found!')
    
    def process_all_batches(self):
        """Process all hourly batches."""
        logger(f'Processing {len(self.processable_hours)} hours', "INFO")
        
        total_processed = 0
        total_time = 0
        
        for i, target_hour_str in enumerate(self.processable_hours):
            batch_start = time.time()
            is_first = (i == 0)
            
            logger(f'Processing {i+1}/{len(self.processable_hours)} - {target_hour_str}', "INFO")
            
            # Load batch
            hour_batch = self.batch_loader.load_batch(self.raw_lob, target_hour_str)
            batch_count = hour_batch.count()
            logger(f'Loaded {batch_count} rows', "INFO")
            
            if batch_count > 0:
                # Process batch
                features_df = self._process_batch(hour_batch, target_hour_str, is_first)
                
                # Write to database
                self._write_to_database(features_df)
                
                processed_count = features_df.count()
                total_processed += processed_count
                features_df.unpersist()
            else:
                logger(f'Skipping empty batch', "WARNING")
            
            # Statistics
            batch_duration = time.time() - batch_start
            total_time += batch_duration
            avg_time = total_time / (i + 1)
            eta = avg_time * (len(self.processable_hours) - i - 1)
            
            logger(f'Batch completed in {batch_duration:.2f}s', "INFO")
            logger(f'Progress: {i+1}/{len(self.processable_hours)}, ETA: {eta:.2f}s', "INFO")
            
            hour_batch.unpersist()
        
        logger(f'Total processed: {total_processed} rows in {total_time:.2f}s', "INFO")
    
    def _process_batch(self, batch: DataFrame, target_hour_str: str, 
                       is_first_batch: bool) -> DataFrame:
        """
        Process a single hourly batch.
        
        Args:
            batch: DataFrame with batch data
            target_hour_str: Target hour string
            is_first_batch: Whether this is the first batch (needs context)
            
        Returns:
            DataFrame with derived features
        """
        logger(f'Deriving features for {target_hour_str} (first={is_first_batch})', "INFO")
        
        # Prepare base dataframe
        base = (
            batch.select("timestamp_str", "bids", "asks", "fold_type", "fold_id", "split_roles")
            .withColumn("trading_hour_str", expr("substring(timestamp_str, 1, 13) || ':00:00.000Z'"))
            .repartition(4)
            .sortWithinPartitions("timestamp_str")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = base.count()
        
        # Calculate price features
        logger('Calculating price features...', "INFO")
        df = self.price_features.calculate_mid_prices(base)
        df = df.select("timestamp_str", "trading_hour_str", "bids", "asks", "mid_price", 
                       "fold_type", "fold_id", "split_roles")
        df = self.price_features.calculate_log_returns(df)
        df = df.select("timestamp_str", "trading_hour_str", "bids", "asks", "mid_price", 
                       "log_return", "fold_type", "fold_id", "split_roles")
        
        # Calculate forward returns (targets)
        logger('Calculating forward returns (targets)...', "INFO")
        df = self.forward_returns.calculate(df)
        
        # Calculate historical returns
        logger('Calculating historical returns...', "INFO")
        df = self.historical_returns.calculate(df)
        
        # Calculate volatility
        logger('Calculating variance and volatility...', "INFO")
        df = self.volatility_features.calculate(df)
        
        # Calculate spread and microprice
        logger('Calculating spread and microprice...', "INFO")
        df = self.price_features.calculate_spread(df)
        df = self.price_features.calculate_microprice(df)
        
        # Calculate depth features
        logger('Calculating depth-based features...', "INFO")
        df = self.depth_features.calculate_all(df)
        
        # Select output columns (keep bids, asks, fold_type, fold_id, split_roles)
        keep = ["timestamp_str", "bids", "asks"]
        keep += [c for c in df.columns if c.startswith("fwd_logret_")]
        keep += [c for c in df.columns if c.startswith("past_logret_")]
        keep += ["mid_price", "volatility", "spread", "microprice"]
        keep += [c for c in df.columns if c.startswith("depth_imbalance_")]
        keep += [c for c in df.columns if c.startswith("depth_") and not c.startswith("depth_imbalance_")]
        keep += [c for c in df.columns if c.startswith("liquidity_concentration_")]
        keep += [c for c in df.columns if c.startswith("price_impact_proxy_")]
        keep += [c for c in df.columns if c.startswith("liquidity_spread_")]
        keep += ["fold_type", "fold_id", "split_roles"]
        
        df = df.select(*keep)
        
        # Filter to appropriate hours
        if is_first_batch:
            # First batch: keep context + target hour
            target_dt = parse_hour_string(target_hour_str)
            context_end_dt = add_hours(target_dt, 1)
            context_end_str = context_end_dt.isoformat() + '.000Z'
            
            df_output = df.filter(col("timestamp_str") < lit(context_end_str))
            total_cnt = df_output.count()
            logger(f'First batch - keeping context + target: {total_cnt} rows', "INFO")
        else:
            # Regular batch: only keep target hour
            df_output = df.filter(
                expr("substring(timestamp_str, 1, 13) || ':00:00.000Z'") == lit(target_hour_str)
            )
            target_cnt = df_output.count()
            logger(f'Regular batch - keeping target hour only: {target_cnt} rows', "INFO")
        
        # Rename timestamp_str back to timestamp for output
        df_output = df_output.withColumnRenamed("timestamp_str", "timestamp")
        
        cnt = df_output.count()
        logger(f'Feature derivation complete - {cnt} rows', "INFO")
        base.unpersist()
        
        return df_output
    
    def _write_to_database(self, df: DataFrame):
        """
        Write processed features to MongoDB.
        
        Args:
            df: DataFrame with features to write
        """
        logger('Writing features to database...', "INFO")
        total = df.count()
        logger(f'Writing {total} rows...', "INFO")
        
        df = df.orderBy("timestamp")
        
        t0 = time.time()
        write_with_timestamp_conversion(
            df,
            self.db_name,
            self.output_collection,
            timestamp_col="timestamp",
            mode="append"
        )
        
        logger(f'Write completed in {time.time() - t0:.2f}s', "INFO")