"""Logging, metrics, and checkpointing utilities."""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class MetricsLogger:
    """Tracks and logs training metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics = []
        self.val_metrics = []
        self.episode_metrics = []
    
    def log_episode(self, split_id: int, episode_idx: int, metrics: Dict):
        """Log episode metrics."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'split_id': split_id,
            'episode_idx': episode_idx,
            **metrics
        }
        self.episode_metrics.append(entry)
    
    def log_epoch(self, epoch: int, split_id: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch-level metrics."""
        train_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'split_id': split_id,
            'phase': 'train',
            **train_metrics
        }
        self.train_metrics.append(train_entry)
        
        val_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'split_id': split_id,
            'phase': 'val',
            **val_metrics
        }
        self.val_metrics.append(val_entry)
    
    def save(self, filename: str = "metrics.json"):
        """Save all metrics to file."""
        metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'episodes': self.episode_metrics
        }
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_best_val_sharpe(self, split_id: int) -> float:
        """Get best validation Sharpe for a split."""
        split_val_metrics = [m for m in self.val_metrics if m['split_id'] == split_id]
        if not split_val_metrics:
            return float('-inf')
        return max(m['sharpe'] for m in split_val_metrics)


def compute_sharpe_ratio(returns: List[float], annualization_factor: float = None) -> float:
    """
    Compute Sharpe ratio from returns.
    
    Args:
        returns: List of returns
        annualization_factor: Factor to annualize (e.g., sqrt(252*48) for 30s data)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    mean_return = returns_array.mean()
    std_return = returns_array.std()
    
    if std_return < 1e-8:
        return 0.0
    
    sharpe = mean_return / std_return
    
    if annualization_factor is not None:
        sharpe *= annualization_factor
    
    return sharpe


def save_checkpoint(
    agent,
    optimizer,
    epoch: int,
    split_id: int,
    val_sharpe: float,
    checkpoint_dir: str = "checkpoints"
):
    """
    Save model checkpoint.
    
    Args:
        agent: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        split_id: Split identifier
        val_sharpe: Validation Sharpe ratio
        checkpoint_dir: Directory for checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'split_id': split_id,
        'val_sharpe': val_sharpe,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = checkpoint_dir / f"agent_split_{split_id}.pth"
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    agent,
    optimizer=None,
    device: str = 'cuda'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        agent: Model to load into
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    agent.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'split_id': checkpoint['split_id'],
        'val_sharpe': checkpoint['val_sharpe'],
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty print metrics."""
    print(f"\n{prefix}")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {key:.<40} {value}")
    print("=" * 60)