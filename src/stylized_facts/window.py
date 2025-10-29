import random
from typing import List, Dict
from datetime import timedelta
from src.utils import logger, log_section

class RepresentativeWindowsExtractor:
    """
    Extracts one representative window per TRAIN fold per split for stylized facts analysis.
    """
    
    def __init__(self, folds: List, splits: List, config: Dict):
        """
        Initializes windows extractor.
        
        Args:
            folds: List of Fold dataclass objects
            splits: List of CPCVsplit objects with train/val fold assignments
            config: Configuration dictionary
        """
        self.folds = folds
        self.splits = splits
        self.config = config
        self.windows: List[Dict] = []
        self.master_seed = config['master_seed']
    
    def extract_windows_per_split(self) -> List[Dict]:
        """
        Extracts one representative window per TRAIN fold for EACH split.
        
        Process:
        - For each CPCV split
        - For each TRAIN fold in that split
        - Extract one random window from the fold's safe zone
        
        Returns:
            List of window metadata dictionaries with split_id and fold assignments
        """
        random.seed(self.master_seed)
        
        L = self.config['stylized_facts']['window_length_samples']
        edge_margin = self.config['stylized_facts']['edge_margin_samples']
        sampling_interval = timedelta(seconds=self.config['temporal_params']['sampling_interval_seconds'])
        min_required = 2 * edge_margin + L
        
        log_section('Extracting Representative Windows for Stylized Facts Analysis')
        logger(f'Window length (L): {L} samples', level="INFO")
        logger(f'Edge margin: {edge_margin} samples on each side', level="INFO")
        logger(f'Minimum required samples per fold: {min_required}', level="INFO")
        logger(f'Number of splits: {len(self.splits)}', level="INFO")
        log_section('', char='-')
        
        # Create fold lookup dictionary (Fold dataclass objects)
        fold_dict = {fold.fold_id: fold for fold in self.folds}
        
        logger(f'Available folds: {sorted(fold_dict.keys())}', level="INFO")
        logger(f'Fold types: {set(f.fold_type for f in self.folds)}', level="INFO")
        
        total_windows = 0
        skipped_too_small = 0
        
        # Process each split (CPCVsplit objects)
        for split in self.splits:
            split_id = split.split_id
            
            # Extract training fold IDs from CPCVsplit object
            train_fold_ids = split.training_folds if hasattr(split, 'training_folds') else []
            
            if not train_fold_ids:
                logger(f'Split {split_id}: No training folds found, skipping', level="WARN")
                continue
            
            logger(f'Split {split_id}: Processing {len(train_fold_ids)} train folds', level="INFO")
            
            # Extract window from each TRAIN fold in this split
            for fold_id in train_fold_ids:
                fold = fold_dict.get(fold_id)
                
                if fold is None:
                    logger(f'  Warning: Fold {fold_id} not found in fold list', level="WARN")
                    continue
                
                # Check if fold has enough samples for safe zone
                if fold.n_samples < min_required:
                    logger(f'  Fold {fold_id}: Only {fold.n_samples} samples, need {min_required}. Skipping.', 
                           level="WARN")
                    skipped_too_small += 1
                    continue
                
                # Calculate safe zone boundaries (indices)
                safe_start_idx = fold.start_idx + edge_margin
                safe_end_idx = fold.end_idx - edge_margin
                safe_zone_size = safe_end_idx - safe_start_idx
                
                # Check if we have enough samples in safe zone for L-length window
                if safe_zone_size < L:
                    logger(f'  Fold {fold_id}: Safe zone only {safe_zone_size} samples, need {L}. Skipping.', 
                           level="WARN")
                    skipped_too_small += 1
                    continue
                
                # Randomly select starting index within safe zone
                # Use split_id and fold_id to create unique random state per window
                window_seed = self.master_seed + split_id * 1000 + fold_id
                random.seed(window_seed)
                
                max_start_position = safe_end_idx - L
                window_start_idx = random.randint(safe_start_idx, max_start_position)
                window_end_idx = window_start_idx + L
                
                # Calculate window timestamps
                samples_from_fold_start = window_start_idx - fold.start_idx
                window_start_ts = fold.start_ts + (sampling_interval * samples_from_fold_start)
                window_end_ts = window_start_ts + (sampling_interval * L)
                
                # Create window metadata
                window = {
                    'split_id': split_id,
                    'fold_id': fold_id,
                    'fold_type': 'train',
                    'window_start_idx': window_start_idx,
                    'window_end_idx': window_end_idx,
                    'window_n_samples': L,
                    'window_start_ts': window_start_ts,
                    'window_end_ts': window_end_ts,
                    'fold_safe_zone': {
                        'start_idx': safe_start_idx,
                        'end_idx': safe_end_idx,
                        'n_samples': safe_zone_size
                    }
                }
                
                self.windows.append(window)
                total_windows += 1
                
                logger(f'  Split {split_id}, Fold {fold_id}: '
                       f'{window_start_ts} to {window_end_ts} '
                       f'[idx {window_start_idx}:{window_end_idx}]', 
                       level="INFO")
        
        logger(f'', level="INFO")
        logger(f'Extracted {total_windows} representative windows across {len(self.splits)} splits', 
               level="INFO")
        if skipped_too_small > 0:
            logger(f'Skipped {skipped_too_small} fold-split combinations (too small)', level="WARN")
        
        # Reset random seed to master seed
        random.seed(self.master_seed)
        
        return self.windows