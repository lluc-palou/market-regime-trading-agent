import random
import numpy as np
from typing import Dict, List

class SinglePassAccumulator:
    """
    Accumulates training and validation data in a single pass through split data.
    
    Maintains separate accumulators for training and validation data per feature,
    applying sampling rate only to training data.
    """
    
    def __init__(self, feature_names: List[str], split_id: int, train_sample_rate: float = 1.0):
        """
        Initialize accumulator.
        
        Args:
            feature_names: List of feature names to accumulate
            split_id: Split ID being processed
            train_sample_rate: Sampling rate for training data (1.0 = all data)
        """
        self.feature_names = feature_names
        self.split_id = split_id
        self.train_sample_rate = train_sample_rate
        
        # Initialize accumulators: {feature_name: [values]}
        self.train_accumulators = {f: [] for f in feature_names}
        self.val_accumulators = {f: [] for f in feature_names}
        
        # Track statistics
        self.samples_processed = 0
        self.hours_processed = 0
    
    def add_sample(self, feature_name: str, role: str, value: float):
        """
        Add a sample to the appropriate accumulator.
        
        Args:
            feature_name: Name of the feature
            role: Role of the sample ('train', 'validation', 'train_warmup')
            value: Feature value
        """
        # Skip non-finite values
        if not np.isfinite(value):
            return
        
        # Route to appropriate accumulator
        if role == 'train':
            # Apply sampling only to training data
            if random.random() <= self.train_sample_rate:
                self.train_accumulators[feature_name].append(value)
        elif role == 'validation':
            # No sampling for validation data
            self.val_accumulators[feature_name].append(value)
        # Ignore other roles (e.g., 'train_warmup')
    
    def get_training_data(self, feature_name: str) -> np.ndarray:
        """
        Get training data for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Numpy array of training samples
        """
        return np.array(self.train_accumulators[feature_name])
    
    def get_validation_data(self, feature_name: str) -> np.ndarray:
        """
        Get validation data for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Numpy array of validation samples
        """
        return np.array(self.val_accumulators[feature_name])
    
    def get_accumulator_stats(self) -> Dict:
        """
        Get statistics about accumulated data.
        
        Returns:
            Dictionary with accumulation statistics
        """
        stats_dict = {
            'split_id': self.split_id,
            'total_samples_processed': self.samples_processed,
            'hours_processed': self.hours_processed,
            'features': {}
        }
        
        for feature_name in self.feature_names:
            stats_dict['features'][feature_name] = {
                'training_samples': len(self.train_accumulators[feature_name]),
                'validation_samples': len(self.val_accumulators[feature_name])
            }
        
        return stats_dict
    
    def clear(self):
        """Clear all accumulated data."""
        self.train_accumulators = {f: [] for f in self.feature_names}
        self.val_accumulators = {f: [] for f in self.feature_names}
        self.samples_processed = 0
        self.hours_processed = 0