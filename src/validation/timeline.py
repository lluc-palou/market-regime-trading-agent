from datetime import datetime
from typing import List, Dict, Tuple, Optional
from src.utils import logger, normalize_timestamp_string

class TimelineAnalyzer:
    """
    Handles time dependent samples management and training/testing sets split with embargo.
    """
    
    def __init__(self, config: Dict, spark, db_name: str):
        """
        Initializes timeline analyzer.
        """
        self.config = config
        self.spark = spark
        self.db_name = db_name
        self.all_timestamps: List = []
        self.usable_timestamps: List[datetime] = []
        self.train_timestamps: List = []
        self.train_warmup_timestamps: List = []
        self.test_timestamps: List = []
        self.test_horizon_timestamps: List = []
        self.embargo_timestamps: List = []
        self.first_usable: Optional[datetime] = None
        self.last_usable: Optional[datetime] = None
    
    def load_timestamps(self, collection: str) -> List[datetime]:
        """
        Loads all DISTINCT timestamps from input database collection as naive datetimes.
        """
        logger(f'Loading distinct timestamps from {collection}...', level="INFO")
        
        # Aggregation pipeline that converts timestamps to strings with millisecond precision
        pipeline = [
            {"$sort": {"timestamp": 1}},
            {"$project": {
                "timestamp": {"$dateToString": {"format": "%Y-%m-%dT%H:%M:%S.%LZ", "date": "$timestamp"}},
                "_id": 0
            }}
        ]
        
        df = (
            self.spark.read.format("mongodb")
            .option("database", self.db_name)
            .option("collection", collection)
            .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
            .load()
        )
        
        # Parse DISTINCT timestamps as naive datetimes
        timestamps = []
        for row in df.select("timestamp").distinct().orderBy("timestamp").collect():
            ts = normalize_timestamp_string(row.timestamp)
            timestamps.append(ts)

        logger(f'Loaded {len(timestamps):,} unique timestamps', level="INFO")
        if timestamps:
            logger(f'Range: {timestamps[0]} to {timestamps[-1]}', level="INFO")
        
        self.all_timestamps = timestamps
        return timestamps
    
    def calculate_usable_range(self) -> Tuple[Optional[datetime], Optional[datetime], List[datetime]]:
        """
        Calculates usable timestamp range.
        """
        logger('Calculating usable range...', level="INFO")
        
        self.first_usable = self.all_timestamps[0]
        self.last_usable = self.all_timestamps[-1]
        self.usable_timestamps = self.all_timestamps.copy()
        
        logger(f'Usable range: {self.first_usable} to {self.last_usable}', level="INFO")
        logger(f'Usable samples: {len(self.usable_timestamps):,}', level="INFO")
        
        return self.first_usable, self.last_usable, self.usable_timestamps
    
    def split_train_test(self) -> Tuple[List[datetime], List[datetime], List[datetime], List[datetime], List[datetime]]:
        """
        Splits usable data into training and testing sets with embargo.
        """
        logger('Splitting usable data into training and testing sets with embargo...', level="INFO")
        
        test_ratio = self.config['train_test_split']['test_ratio']
        embargo_samples = self.config['temporal_params']['embargo_length_samples']
        warmup_samples = self.config['temporal_params']['context_length_samples']
        horizon_samples = self.config['temporal_params']['forecast_horizon_steps']
        n_usable = len(self.usable_timestamps)
        
        # Calculate splitting point
        test_start_idx = int(n_usable * (1 - test_ratio))
        
        # Define embargoed zone
        embargo_start_idx = max(0, test_start_idx - embargo_samples)  
        train_timestamps_initial = self.usable_timestamps[:embargo_start_idx]
        self.embargo_timestamps = self.usable_timestamps[embargo_start_idx:test_start_idx]
        test_timestamps_initial = self.usable_timestamps[test_start_idx:]
        
        # Extract warmup from training set
        if len(train_timestamps_initial) > warmup_samples:
            self.train_warmup_timestamps = train_timestamps_initial[:warmup_samples]
            self.train_timestamps = train_timestamps_initial[warmup_samples:]
            
            logger(f'Training warmup samples: {len(self.train_warmup_timestamps):,} ({len(self.train_warmup_timestamps)/n_usable*100:.1f}%)', level="INFO")
            logger(f'Time range: {self.train_warmup_timestamps[0]} to {self.train_warmup_timestamps[-1]}', level="INFO")
            logger(f'Usable training samples (after warmup): {len(self.train_timestamps):,} ({len(self.train_timestamps)/n_usable*100:.1f}%)', level="INFO")
        else:
            logger(f'Warning: Not enough training samples for warmup period', level="WARN")
            self.train_warmup_timestamps = []
            self.train_timestamps = train_timestamps_initial
            logger(f'Training samples: {len(self.train_timestamps):,} ({len(self.train_timestamps)/n_usable*100:.1f}%)', level="INFO")
            if self.train_timestamps:
                logger(f'Time range: {self.train_timestamps[0]} to {self.train_timestamps[-1]}', level="INFO")
        
        # Extract horizon samples from test set
        if len(test_timestamps_initial) > horizon_samples:
            self.test_timestamps = test_timestamps_initial[:-horizon_samples]
            self.test_horizon_timestamps = test_timestamps_initial[-horizon_samples:]
            
            logger(f'Usable test samples (after horizon): {len(self.test_timestamps):,} ({len(self.test_timestamps)/n_usable*100:.1f}%)', level="INFO")
            logger(f'Time range: {self.test_timestamps[0]} to {self.test_timestamps[-1]}', level="INFO")
            logger(f'Test horizon samples: {len(self.test_horizon_timestamps):,} ({len(self.test_horizon_timestamps)/n_usable*100:.1f}%)', level="INFO")
            logger(f'Time range: {self.test_horizon_timestamps[0]} to {self.test_horizon_timestamps[-1]}', level="INFO")

        else:
            logger(f'Warning: Not enough test samples for horizon period', level="WARN")
            self.test_timestamps = test_timestamps_initial
            self.test_horizon_timestamps = []
            if self.test_timestamps:
                logger(f'Test samples: {len(self.test_timestamps):,} ({len(self.test_timestamps)/n_usable*100:.1f}%)', level="INFO")
                logger(f'Time range: {self.test_timestamps[0]} to {self.test_timestamps[-1]}', level="INFO")
        
        logger(f'Embargo samples: {len(self.embargo_timestamps):,} ({len(self.embargo_timestamps)/n_usable*100:.1f}%)', level="INFO")
        if self.embargo_timestamps:
            logger(f'Time range: {self.embargo_timestamps[0]} to {self.embargo_timestamps[-1]}', level="INFO")
        
        return self.train_warmup_timestamps, self.train_timestamps, self.embargo_timestamps, self.test_timestamps, self.test_horizon_timestamps