"""Episode management and data loading."""

import torch
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Iterator
from collections import defaultdict


class Episode:
    """Represents one trading day episode."""
    
    def __init__(self, split_id: int, date: datetime.date, samples: List[Dict]):
        self.split_id = split_id
        self.date = date
        self.samples = samples
        self.length = len(samples)
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        return f"Episode(split={self.split_id}, date={self.date}, samples={self.length})"


class EpisodeLoader:
    """Loads and manages episodes from MongoDB."""
    
    def __init__(self, config):
        self.config = config
        self.client = MongoClient(config.mongodb_uri)
        self.db = self.client[config.database_name]
    
    def load_episodes(
        self,
        split_id: int,
        role: str = 'train'
    ) -> List[Episode]:
        """
        Load episodes for a split.
        
        Args:
            split_id: Split identifier
            role: 'train' or 'val'
        
        Returns:
            List of Episode objects
        """
        collection = self.db[f'split_{split_id}']
        
        # Query samples with role filter, sorted by timestamp
        cursor = collection.find(
            {'role': role},
            sort=[('timestamp', 1)]
        )
        
        # Group samples by calendar day
        episodes_by_date = defaultdict(list)
        current_fold_id = None
        
        for doc in cursor:
            # Check for fold boundary
            if current_fold_id is not None and doc['fold_id'] != current_fold_id:
                # Fold changed, may need to end current episode early
                pass
            current_fold_id = doc['fold_id']
            
            # Extract date
            timestamp = doc['timestamp']
            if isinstance(timestamp, datetime):
                date = timestamp.date()
            else:
                # Unix timestamp
                date = datetime.fromtimestamp(timestamp).date()
            
            # Prepare sample
            sample = {
                'codebook': doc['codebook'],
                'features': torch.tensor(doc['features'], dtype=torch.float32),
                'timestamp': timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp,
                'target': doc['target'],
                'fold_id': doc['fold_id']
            }
            
            episodes_by_date[date].append(sample)
        
        # Create Episode objects
        episodes = []
        for date in sorted(episodes_by_date.keys()):
            samples = episodes_by_date[date]
            
            # Check for fold boundaries within day
            fold_ids = [s['fold_id'] for s in samples]
            if len(set(fold_ids)) > 1:
                # Fold boundary within day, split into separate episodes
                current_fold = fold_ids[0]
                current_samples = []
                
                for sample in samples:
                    if sample['fold_id'] != current_fold:
                        # New fold, save current episode
                        if current_samples:
                            episodes.append(Episode(split_id, date, current_samples))
                        current_samples = [sample]
                        current_fold = sample['fold_id']
                    else:
                        current_samples.append(sample)
                
                # Save last episode
                if current_samples:
                    episodes.append(Episode(split_id, date, current_samples))
            else:
                # Single fold, create one episode
                episodes.append(Episode(split_id, date, samples))
        
        return episodes
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def get_valid_timesteps(episode: Episode, window_size: int, horizon: int) -> range:
    """
    Get valid timestep range for an episode.
    
    Args:
        episode: Episode object
        window_size: Observation window W
        horizon: Reward horizon H
    
    Returns:
        Range of valid timesteps [W, T-H]
    """
    T = len(episode)
    start = window_size  # Need W samples for context
    end = T - horizon    # Need H samples for reward
    
    if end <= start:
        # Episode too short
        return range(0, 0)
    
    return range(start, end)