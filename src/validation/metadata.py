import yaml
from typing import List, Dict
from datetime import datetime
from src.utils import logger, utcnow, format_timestamp_for_metadata

class MetadataHandler:
    """
    Handles method summary creation as metadata for reproducibility purposes.
    """
    
    def __init__(self, config: Dict):
        """
        Initializes metadata handler.
        """
        self.config = config
    
    def create_metadata(self, analyzer, folds: List, cpcv_splits: List, representative_windows: List[Dict]) -> Dict:
        """
        Retrieves method metadata and returns a summary.
        """
        logger('Retrieving metadata...', level="INFO")
        
        metadata = {
            'experiment_id': self.config['experiment_id'],
            'master_seed': self.config['master_seed'],
            'created_at': format_timestamp_for_metadata(utcnow()),
            
            'data_summary': {
                'total_samples': len(analyzer.all_timestamps),
                'timestamp_range': {
                    'start': analyzer.all_timestamps[0],
                    'end': analyzer.all_timestamps[-1]
                },
                'usable_samples': len(analyzer.usable_timestamps),
                'train_warmup_samples': len(analyzer.train_warmup_timestamps),
                'train_samples': len(analyzer.train_timestamps),
                'embargoed_samples': len(analyzer.embargo_timestamps),
                'test_samples': len(analyzer.test_timestamps),
                'test_horizon_samples': len(analyzer.test_horizon_timestamps)
            },
            
            'temporal_config': self.config['temporal_params'],
            'train_test_config': self.config['train_test_split'],
            'cpcv_config': self.config['cpcv'],
            'stylized_facts_config': self.config['stylized_facts'],
            
            'folds': [fold.to_dict() for fold in folds],
            'cpcv_splits': [split.to_dict() for split in cpcv_splits],
            'split_count': len(cpcv_splits),
            
            'stylized_facts_windows': {
                'window_length_samples': self.config['stylized_facts']['window_length_samples'],
                'edge_margin_samples': self.config['stylized_facts']['edge_margin_samples'],
                'windows': representative_windows,
                'total_windows': len(representative_windows)
            }
        }
        
        return metadata
    
    def write_metadata(self, metadata: Dict, filepath: str) -> None:
        """
        Writes metadata summary to YAML file.
        """
        logger(f'Writing metadata to {filepath}...', level="INFO")
        
        metadata_serializable = self._convert_datetimes(metadata)
        
        with open(filepath, 'w') as f:
            yaml.dump(metadata_serializable, f, default_flow_style=False, sort_keys=False)
        
        logger(f'Metadata writing complete.', level="INFO")
    
    def _convert_datetimes(self, obj):
        """
        Converts datetimes to ISO format with explicit UTC marker.
        """
        if isinstance(obj, datetime):
            return format_timestamp_for_metadata(obj)
        
        elif isinstance(obj, dict):
            return {k: self._convert_datetimes(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._convert_datetimes(item) for item in obj]
        
        else:
            return obj