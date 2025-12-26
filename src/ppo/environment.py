"""Episode management and data loading."""

import torch
from pymongo import MongoClient, ASCENDING
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
        self.client = MongoClient(config.mongodb_uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[config.database_name]
        self.is_synthetic = (config.experiment_type.value == 4)  # Experiment 4 uses synthetic data
        self._ensure_indexes()
    
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
        # Route to synthetic loader for Experiment 4
        if self.is_synthetic:
            return self._load_synthetic_episodes(split_id, role)

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
        collection = self.db[f'split_{split_id}_input']  # Match VQVAE output naming convention
        
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
                'codebook': doc['codebook'],  # VQVAE writes 'codebook' field
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

    def _load_synthetic_episodes(
        self,
        split_id: int,
        role: str
    ) -> List[Episode]:
        """
        Load episodes from synthetic data (Experiment 4).

        Synthetic data is organized by sequence_id (120 samples each).
        For training, use first 80% of sequences; for validation, use last 20%.

        Args:
            split_id: Split identifier
            role: 'train' or 'val'

        Returns:
            List of Episode objects
        """
        collection = self.db[f'split_{split_id}_synthetic']

        # Query all synthetic samples sorted by sequence_id and position
        cursor = collection.find(
            {'is_synthetic': True},
            sort=[('sequence_id', 1), ('position_in_sequence', 1)]
        )

        # Group samples by sequence_id
        sequences_by_id = defaultdict(list)

        for doc in cursor:
            sequence_id = doc['sequence_id']

            # Prepare sample
            sample = {
                'codebook': doc['codebook_index'],  # Note: field is 'codebook_index' in synthetic
                'features': None,  # No features in synthetic data
                'timestamp': doc['timestamp'].timestamp() if isinstance(doc['timestamp'], datetime) else doc['timestamp'],
                'target': 0.0,  # Placeholder - synthetic data has no targets
                'fold_id': 0,  # No fold concept in synthetic data
                'position_in_sequence': doc['position_in_sequence']
            }

            sequences_by_id[sequence_id].append(sample)

        # Split sequences into train/val
        sequence_ids = sorted(sequences_by_id.keys())
        n_sequences = len(sequence_ids)
        split_idx = int(0.8 * n_sequences)  # 80/20 split

        if role == 'train':
            selected_ids = sequence_ids[:split_idx]
        else:  # validation
            selected_ids = sequence_ids[split_idx:]

        # Create episodes (each sequence is an episode)
        episodes = []
        for seq_id in selected_ids:
            samples = sequences_by_id[seq_id]

            # Sort by position_in_sequence to ensure correct order
            samples.sort(key=lambda s: s['position_in_sequence'])

            # Remove position_in_sequence field (not needed after sorting)
            for sample in samples:
                del sample['position_in_sequence']

            # Use sequence_id as "date" for Episode
            from datetime import date
            synthetic_date = date(2024, 1, 1)  # Placeholder date

            episodes.append(Episode(split_id, synthetic_date, samples))

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