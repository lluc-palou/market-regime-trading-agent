from src.utils import logger
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Fold:
    """
    A batch of data samples ordered in time (CPCV fold).
    """
    fold_id: int
    start_ts: datetime
    end_ts: datetime
    start_idx: int
    end_idx: int
    fold_type: str  # 'train', 'train_warmup', 'test', 'test_horizon', 'train_test_embargo'
    
    @property
    def n_samples(self) -> int:
        return self.end_idx - self.start_idx
    
    def to_dict(self) -> Dict:
        """Returns dictionary representation of the fold."""
        return {
            'fold_id': self.fold_id,
            'start_ts': self.start_ts,
            'end_ts': self.end_ts,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'n_samples': self.n_samples,
            'fold_type': self.fold_type
        }

class FoldsDivider:
    """
    Divides usable data into CPCV folds.
    """
    
    def __init__(self, train_warmup_timestamps: List[datetime], train_timestamps: List[datetime],
                 embargo_timestamps: List[datetime], test_timestamps: List[datetime], 
                 test_horizon_timestamps: List[datetime], config: Dict):
        self.train_warmup_timestamps = train_warmup_timestamps
        self.train_timestamps = train_timestamps
        self.embargo_timestamps = embargo_timestamps
        self.test_timestamps = test_timestamps
        self.test_horizon_timestamps = test_horizon_timestamps
        self.config = config
        self.folds: List[Fold] = []
    
    def divide_timeline(self) -> List[Fold]:
        """
        Divides all usable data into training, warmup, embargoed, test and horizon folds.
        """
        n_folds = self.config['cpcv']['n_folds']
        total_train_samples = len(self.train_timestamps)
        fold_size = total_train_samples // n_folds
        
        logger(f'Dividing {total_train_samples:,} training samples into {n_folds} folds', level="INFO")
        
        current_idx = 0
        current_fold_id = 0
        
        # Create warmup fold FIRST
        if self.train_warmup_timestamps:
            warmup_end_boundary = self.train_timestamps[0]
            
            warmup_fold = Fold(
                fold_id=current_fold_id,
                start_ts=self.train_warmup_timestamps[0],
                end_ts=warmup_end_boundary,
                start_idx=current_idx,
                end_idx=current_idx + len(self.train_warmup_timestamps),
                fold_type='train_warmup'
            )
            
            self.folds.append(warmup_fold)
            duration_hours = (warmup_fold.end_ts - warmup_fold.start_ts).total_seconds() / 3600
            logger(f'Fold {current_fold_id} (warmup): {warmup_fold.start_ts} to {warmup_fold.end_ts} ({warmup_fold.n_samples:,} samples, ~{duration_hours:.1f} hours)', level="INFO")
            
            current_idx += len(self.train_warmup_timestamps)
            current_fold_id += 1
        
        # Create training folds
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else total_train_samples
            
            # Calculate proper end boundary
            if i < n_folds - 1:
                next_fold_start_idx = (i + 1) * fold_size
                fold_end_ts = self.train_timestamps[next_fold_start_idx]
            else:
                if self.embargo_timestamps:
                    fold_end_ts = self.embargo_timestamps[0]
                elif self.test_timestamps:
                    fold_end_ts = self.test_timestamps[0]
                else:
                    fold_end_ts = self.train_timestamps[-1] + timedelta(days=1)
            
            fold = Fold(
                fold_id=current_fold_id,
                start_ts=self.train_timestamps[start_idx],
                end_ts=fold_end_ts,
                start_idx=current_idx + start_idx,
                end_idx=current_idx + end_idx,
                fold_type='train'
            )
            
            self.folds.append(fold)
            duration_hours = (fold.end_ts - fold.start_ts).total_seconds() / 3600
            logger(f'Fold {current_fold_id} (train): {fold.start_ts} to {fold.end_ts} ({fold.n_samples:,} samples, ~{duration_hours:.1f} hours)', level="INFO")
            
            current_fold_id += 1
        
        current_idx += total_train_samples
        
        # Create embargo fold
        if self.embargo_timestamps:
            embargo_end_boundary = self.test_timestamps[0] if self.test_timestamps else self.embargo_timestamps[-1] + timedelta(days=1)
            
            embargoed_fold = Fold(
                fold_id=current_fold_id,
                start_ts=self.embargo_timestamps[0],
                end_ts=embargo_end_boundary,
                start_idx=current_idx,
                end_idx=current_idx + len(self.embargo_timestamps),
                fold_type='train_test_embargo'
            )
            
            self.folds.append(embargoed_fold)
            duration_hours = (embargoed_fold.end_ts - embargoed_fold.start_ts).total_seconds() / 3600
            logger(f'Fold {current_fold_id} (embargoed): {embargoed_fold.start_ts} to {embargoed_fold.end_ts} ({embargoed_fold.n_samples:,} samples, ~{duration_hours:.1f} hours)', level="INFO")
            
            current_idx += len(self.embargo_timestamps)
            current_fold_id += 1
        
        # Create test fold
        if self.test_timestamps:
            test_end_boundary = self.test_horizon_timestamps[0] if self.test_horizon_timestamps else self.test_timestamps[-1] + timedelta(days=1)
            
            test_fold = Fold(
                fold_id=current_fold_id,
                start_ts=self.test_timestamps[0],
                end_ts=test_end_boundary,
                start_idx=current_idx,
                end_idx=current_idx + len(self.test_timestamps),
                fold_type='test'
            )
            
            self.folds.append(test_fold)
            duration_hours = (test_fold.end_ts - test_fold.start_ts).total_seconds() / 3600
            logger(f'Fold {current_fold_id} (test): {test_fold.start_ts} to {test_fold.end_ts} ({test_fold.n_samples:,} samples, ~{duration_hours:.1f} hours)', level="INFO")
            
            current_idx += len(self.test_timestamps)
            current_fold_id += 1
        
        # Create test horizon fold
        if self.test_horizon_timestamps:
            horizon_end_boundary = self.test_horizon_timestamps[-1] + timedelta(days=1)
            
            horizon_fold = Fold(
                fold_id=current_fold_id,
                start_ts=self.test_horizon_timestamps[0],
                end_ts=horizon_end_boundary,
                start_idx=current_idx,
                end_idx=current_idx + len(self.test_horizon_timestamps),
                fold_type='test_horizon'
            )
            
            self.folds.append(horizon_fold)
            duration_hours = (horizon_fold.end_ts - horizon_fold.start_ts).total_seconds() / 3600
            logger(f'Fold {current_fold_id} (horizon): {horizon_fold.start_ts} to {horizon_fold.end_ts} ({horizon_fold.n_samples:,} samples, ~{duration_hours:.1f} hours)', level="INFO")
        
        return self.folds