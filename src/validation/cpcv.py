from src.utils import logger
from itertools import combinations
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

class CPCVsplit:
    """
    Represents one fold configuration in the CPCV method (one iteration).
    """
    
    def __init__(self, split_id: int, validation_folds: List[int], training_folds: List[int]):
        """
        Initializes CPCV split.
        """
        self.split_id = split_id
        self.validation_folds = validation_folds
        self.training_folds = training_folds
        self.purged_ranges: Dict[int, List[Tuple[datetime, datetime]]] = {}
        self.embargoed_ranges: Dict[int, List[Tuple[datetime, datetime]]] = {}
    
    def set_purge_embargo_ranges(self, purged_ranges: Dict[int, List[Tuple[datetime, datetime]]], 
                                  embargoed_ranges: Dict[int, List[Tuple[datetime, datetime]]]):
        """
        Defines which timestamp ranges from the split will be purged or embargoed.
        """
        self.purged_ranges = purged_ranges
        self.embargoed_ranges = embargoed_ranges
    
    def to_dict(self) -> Dict:
        """
        Returns dictionary representation of the split.
        """
        return {
            'split_id': self.split_id,
            'validation_folds': self.validation_folds,
            'training_folds': self.training_folds,
            'purged_ranges': {k: [[t[0], t[1]] for t in v] for k, v in self.purged_ranges.items()},
            'embargoed_ranges': {k: [[t[0], t[1]] for t in v] for k, v in self.embargoed_ranges.items()}
        }


class CPCVsplitGenerator:
    """
    Generates CPCV splits with sample-level purge and embargo using timestamp ranges.
    """
    
    def __init__(self, folds: List, train_timestamps: List[datetime], config: Dict):
        """
        Initializes CPCV split generator.
        """
        self.folds = [f for f in folds if f.fold_type == 'train']
        self.folds_by_id = {f.fold_id: f for f in self.folds}
        self.train_timestamps = train_timestamps
        self.config = config
        self.splits: List[CPCVsplit] = []
    
    def generate_splits(self) -> List[CPCVsplit]:
        """
        Generates all possible combinations of folds (splits).
        """
        n_folds = len(self.folds)
        k_val_folds = self.config['cpcv']['k_validation_folds']
        all_combinations = list(combinations(range(n_folds), k_val_folds))
    
        logger(f'Generating C({n_folds},{k_val_folds}) = {len(all_combinations)} CPCV splits...', level="INFO")
        
        purge_samples = self.config['temporal_params']['purge_length_samples']
        embargo_samples = self.config['temporal_params']['embargo_length_samples']
        
        for split_id, val_idx_tuple in enumerate(all_combinations):
            val_idx = list(val_idx_tuple)
            train_idx = [i for i in range(n_folds) if i not in val_idx]

            # Map list indices to true fold_ids
            val_folds_ids = [self.folds[i].fold_id for i in val_idx]
            train_folds_ids = [self.folds[i].fold_id for i in train_idx]

            split = CPCVsplit(split_id, val_folds_ids, train_folds_ids)

            # Calculate purged and embargoed timestamp ranges
            purged_ranges, embargoed_ranges = self.calculate_purged_embargoed_ranges(
                val_folds_ids, purge_samples, embargo_samples
            )

            split.set_purge_embargo_ranges(purged_ranges, embargoed_ranges)
            self.splits.append(split)
            
            total_purged = sum(len(v) for v in purged_ranges.values())
            total_embargoed = sum(len(v) for v in embargoed_ranges.values())

            logger(f'CPCV split {split_id}: val={val_folds_ids}, train={train_folds_ids}, '
                   f'purged_ranges={total_purged}, embargoed_ranges={total_embargoed}', level="INFO")
        
        logger(f'Generated {len(self.splits)} CPCV splits.', level="INFO")

        return self.splits
    
    def calculate_purged_embargoed_ranges(self, val_folds_ids: List[int], 
                                          purge_samples: int, 
                                          embargo_samples: int) -> Tuple[Dict[int, List[Tuple[datetime, datetime]]], 
                                                                          Dict[int, List[Tuple[datetime, datetime]]]]:
        """
        Calculates timestamp ranges that should be purged or embargoed for given validation folds.
        """
        sampling_interval = timedelta(seconds=self.config['temporal_params']['sampling_interval_seconds'])
        
        purged_ranges: Dict[int, List[Tuple[datetime, datetime]]] = {}
        embargoed_ranges: Dict[int, List[Tuple[datetime, datetime]]] = {}
        
        for val_fold_id in val_folds_ids:
            val_fold = self.folds_by_id[val_fold_id]
            
            # Calculate purge timestamp range (BEFORE validation fold)
            purge_duration = sampling_interval * purge_samples
            purge_start_ts = val_fold.start_ts - purge_duration
            purge_end_ts = val_fold.start_ts
            
            # Calculate embargo timestamp range (AFTER validation fold)
            embargo_duration = sampling_interval * embargo_samples
            embargo_start_ts = val_fold.end_ts
            embargo_end_ts = val_fold.end_ts + embargo_duration
            
            # Find which training folds overlap with purge/embargo ranges
            for fold in self.folds_by_id.values():
                if fold.fold_id in val_folds_ids:
                    continue
                
                # Check purge overlap
                if fold.start_ts < purge_end_ts and fold.end_ts > purge_start_ts:
                    overlap_start = max(fold.start_ts, purge_start_ts)
                    overlap_end = min(fold.end_ts, purge_end_ts)
                    
                    if fold.fold_id not in purged_ranges:
                        purged_ranges[fold.fold_id] = []
                    purged_ranges[fold.fold_id].append((overlap_start, overlap_end))
                
                # Check embargo overlap
                if fold.start_ts < embargo_end_ts and fold.end_ts > embargo_start_ts:
                    overlap_start = max(fold.start_ts, embargo_start_ts)
                    overlap_end = min(fold.end_ts, embargo_end_ts)
                    
                    if fold.fold_id not in embargoed_ranges:
                        embargoed_ranges[fold.fold_id] = []
                    embargoed_ranges[fold.fold_id].append((overlap_start, overlap_end))
        
        return purged_ranges, embargoed_ranges