"""Episode management and data loading."""

import torch
from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Iterator
from collections import defaultdict


class Episode:
    """Represents one trading episode (can be a day or hourly chunk)."""

    def __init__(self, split_id: int, date: datetime.date, samples: List[Dict], parent_id: Optional[int] = None, chunk_id: Optional[int] = None):
        self.split_id = split_id
        self.date = date
        self.samples = samples
        self.length = len(samples)
        self.parent_id = parent_id  # ID of parent day-episode (for hourly chunks)
        self.chunk_id = chunk_id    # Chunk number within parent episode (0, 1, 2, ...)

    def __len__(self):
        return self.length

    def __repr__(self):
        if self.parent_id is not None:
            return f"Episode(split={self.split_id}, date={self.date}, samples={self.length}, parent={self.parent_id}, chunk={self.chunk_id})"
        return f"Episode(split={self.split_id}, date={self.date}, samples={self.length})"


class EpisodeLoader:
    """Loads and manages episodes from MongoDB."""

    def __init__(self, config, episode_chunk_size: int = 120):
        self.config = config
        self.client = MongoClient(config.mongodb_uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[config.database_name]
        self.experiment_type = config.experiment_type
        self.episode_chunk_size = episode_chunk_size  # Split episodes into chunks of this size (e.g., 120 for 1 hour)
        self._ensure_indexes()
    
    def load_episodes(
        self,
        split_id: int,
        role: str = 'train'
    ) -> List[Episode]:
        """
        Load episodes for a split from original data.

        Args:
            split_id: Split identifier
            role: 'train' or 'val'

        Returns:
            List of Episode objects
        """
        return self._load_original_episodes(split_id, role)

    def _load_original_episodes(
        self,
        split_id: int,
        role: str
    ) -> List[Episode]:
        """
        Load episodes from original data (Experiments 1, 2, 3).

        Args:
            split_id: Split identifier
            role: 'train' or 'val'

        Returns:
            List of Episode objects
        """
        # Map 'val' to 'validation' for database query (DB uses 'validation', not 'val')
        db_role = 'validation' if role == 'val' else role

        collection = self.db[f'split_{split_id}_input']  # Match VQVAE output naming convention

        # Query samples with role filter, sorted by timestamp
        cursor = collection.find(
            {'role': db_role},
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
                'codebook': doc['codebook_index'],  # Field name is 'codebook_index' in database
                'features': torch.tensor(doc['features'], dtype=torch.float32),
                'timestamp': timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp,
                'target': doc['target'],
                'fold_id': doc['fold_id']
            }
            
            episodes_by_date[date].append(sample)
        
        # Create Episode objects (day-level first)
        day_episodes = []
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
                            day_episodes.append(Episode(split_id, date, current_samples))
                        current_samples = [sample]
                        current_fold = sample['fold_id']
                    else:
                        current_samples.append(sample)

                # Save last episode
                if current_samples:
                    day_episodes.append(Episode(split_id, date, current_samples))
            else:
                # Single fold, create one episode
                day_episodes.append(Episode(split_id, date, samples))

        # Split each day episode into hourly chunks
        episodes = []
        parent_id = 0
        for day_episode in day_episodes:
            if len(day_episode) <= self.episode_chunk_size:
                # Episode short enough, use as-is
                episodes.append(day_episode)
                parent_id += 1
            else:
                # Split into chunks
                num_chunks = (len(day_episode) + self.episode_chunk_size - 1) // self.episode_chunk_size
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * self.episode_chunk_size
                    end_idx = min(start_idx + self.episode_chunk_size, len(day_episode))
                    chunk_samples = day_episode.samples[start_idx:end_idx]

                    # Create chunk episode with parent tracking
                    chunk_episode = Episode(
                        split_id, day_episode.date, chunk_samples,
                        parent_id=parent_id, chunk_id=chunk_idx
                    )
                    episodes.append(chunk_episode)
                parent_id += 1

        return episodes

    def _ensure_indexes(self):
        """
        Ensure timestamp indexes exist on all split collections.

        Follows the pattern from other pipeline stages (03-14) for efficient
        timestamp-based queries.
        """
        for split_id in self.config.split_ids:
            collection_name = f'split_{split_id}_input'
            if collection_name in self.db.list_collection_names():
                collection = self.db[collection_name]

                # Check if timestamp index exists
                existing_indexes = list(collection.list_indexes())
                has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

                if not has_timestamp_index:
                    # Create index for efficient timestamp-based queries
                    collection.create_index([("timestamp", ASCENDING)], background=False)

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