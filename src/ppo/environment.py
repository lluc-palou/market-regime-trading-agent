"""Episode management and data loading."""

import torch
from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Iterator
from collections import defaultdict
from src.utils.logging import logger


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
    """Loads and manages episodes from MongoDB with optional pre-tensorization."""

    def __init__(self, config, episode_chunk_size: int = 120, pre_tensorize: bool = True, use_pinned_memory: bool = True):
        self.config = config
        self.client = MongoClient(config.mongodb_uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[config.database_name]
        self.experiment_type = config.experiment_type
        self.episode_chunk_size = episode_chunk_size  # Split episodes into chunks of this size (e.g., 120 for 1 hour)
        self.pre_tensorize = pre_tensorize  # Convert all data to tensors during loading
        self.use_pinned_memory = use_pinned_memory  # Use pinned memory for faster GPU transfer
        self._ensure_indexes()
    
    def load_episodes(
        self,
        split_id: int,
        role: str = 'train'
    ) -> List[Episode]:
        """
        Load episodes for a split (original or synthetic based on experiment type).

        Args:
            split_id: Split identifier
            role: 'train', 'val', or None (None loads all roles)

        Returns:
            List of Episode objects
        """
        from .config import ExperimentType

        # Experiment 4 uses synthetic data, others use original
        if self.experiment_type == ExperimentType.EXP4_SYNTHETIC_BINS:
            return self._load_synthetic_episodes(split_id, role)
        else:
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
            role: 'train', 'val', or None (None loads all roles)

        Returns:
            List of Episode objects
        """
        collection = self.db[f'split_{split_id}_input']  # Match VQVAE output naming convention

        # Build query filter
        if role is None:
            # Load all roles
            query_filter = {}
        else:
            # Map 'val' to 'validation' for database query (DB uses 'validation', not 'val')
            db_role = 'validation' if role == 'val' else role
            query_filter = {'role': db_role}

        # Query samples with role filter, sorted by timestamp
        cursor = collection.find(
            query_filter,
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
            
            # Prepare sample (defer tensorization until later for efficiency)
            sample = {
                'codebook': doc['codebook_index'],  # Field name is 'codebook_index' in database
                'features': doc['features'],  # Keep as list for now
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

        # Tensorize all episode samples if enabled (batch operation for efficiency)
        if self.pre_tensorize:
            episodes = self._tensorize_episodes(episodes)

        return episodes

    def _tensorize_episodes(self, episodes: List[Episode]) -> List[Episode]:
        """
        Convert all episode sample data to tensors with optional pinned memory.
        This is done in batch after loading for better performance.
        """
        logger(f"  Pre-tensorizing {len(episodes)} episodes...", "INFO")

        for episode in episodes:
            for sample in episode.samples:
                # Convert features to tensor (skip if None for codebook-only experiments)
                if sample['features'] is not None and not isinstance(sample['features'], torch.Tensor):
                    features_tensor = torch.tensor(sample['features'], dtype=torch.float32)
                    if self.use_pinned_memory:
                        features_tensor = features_tensor.pin_memory()
                    sample['features'] = features_tensor

        logger(f"  ✓ Episodes tensorized", "INFO")
        return episodes

    def _load_synthetic_episodes(
        self,
        split_id: int,
        role: str
    ) -> List[Episode]:
        """
        Load episodes from synthetic data (Experiment 4).

        Synthetic data is organized by sequence_id (100 sequences per split),
        with each sequence containing 120 samples ordered by position_in_sequence.

        Since synthetic data has NO 'role' field, we split by sequence_id:
        - Train: sequences 0-79 (80 sequences, 80%)
        - Validation: sequences 80-99 (20 sequences, 20%)

        Args:
            split_id: Split identifier
            role: 'train', 'val', or None (None loads all roles)

        Returns:
            List of Episode objects (one per sequence)
        """
        collection = self.db[f'split_{split_id}_synthetic']  # Synthetic data collection

        # Build query filter based on sequence_id (synthetic data has no 'role' field)
        if role is None:
            # Load all sequences (0-99)
            query_filter = {}
        elif role == 'train':
            # Train: sequences 0-79 (80%)
            query_filter = {'sequence_id': {'$lt': 80}}
        else:  # role == 'val' or 'validation'
            # Validation: sequences 80-99 (20%)
            query_filter = {'sequence_id': {'$gte': 80}}

        # Query synthetic samples with sequence filter, sorted by sequence and position
        cursor = collection.find(
            query_filter,
            sort=[('sequence_id', 1), ('position_in_sequence', 1)]
        )

        # Group samples by sequence_id
        episodes_by_sequence = defaultdict(list)

        for doc in cursor:
            sequence_id = doc['sequence_id']

            # Extract and convert timestamp to Unix timestamp (float)
            timestamp = doc.get('timestamp', 0)
            if isinstance(timestamp, datetime):
                timestamp = timestamp.timestamp()

            # Prepare sample (defer tensorization until later for efficiency)
            # Synthetic data: Uses same feature source as Experiment 3 (codebook indices only)
            sample = {
                'codebook': doc['codebook_index'],  # Codebook index (0-127)
                'features': None,  # No features for codebook-only experiment
                'timestamp': timestamp,  # Convert datetime to Unix timestamp
                'target': doc['target'],
                'sequence_id': sequence_id,
                'position_in_sequence': doc['position_in_sequence']
            }

            episodes_by_sequence[sequence_id].append(sample)

        # Create Episode objects (one per sequence)
        # Sequences are already 120 samples, so no chunking needed
        episodes = []
        for sequence_id in sorted(episodes_by_sequence.keys()):
            samples = episodes_by_sequence[sequence_id]

            # Verify sequence length
            if len(samples) != 120:
                logger(f"Warning: Sequence {sequence_id} has {len(samples)} samples (expected 120)", "WARNING")

            # Use sequence_id as the "date" identifier for Episode
            # This keeps Episode API consistent while using sequence-based organization
            from datetime import date as dt_date
            synthetic_date = dt_date(2000, 1, 1)  # Dummy date for synthetic data

            episode = Episode(split_id, synthetic_date, samples, parent_id=sequence_id, chunk_id=0)
            episodes.append(episode)

        # Tensorize all episode samples if enabled (batch operation for efficiency)
        if self.pre_tensorize:
            episodes = self._tensorize_episodes(episodes)

        return episodes

    def load_test_episodes(self, collection_name: str = 'test_data') -> List[Episode]:
        """
        Load episodes from test_data collection for final evaluation.

        This method loads ALL episodes from the test collection (no role filter)
        for final model evaluation.

        Args:
            collection_name: Name of test collection (default: 'test_data')

        Returns:
            List of Episode objects from test data
        """
        logger(f'Loading test episodes from {collection_name}...', "INFO")

        collection = self.db[collection_name]

        # Query all samples (no role filter for test data), sorted by timestamp
        cursor = collection.find(
            {},
            sort=[('timestamp', 1)]
        )

        # Group samples by calendar day
        episodes_by_date = defaultdict(list)

        for doc in cursor:
            # Extract date
            timestamp = doc['timestamp']
            if isinstance(timestamp, datetime):
                date = timestamp.date()
            else:
                # Unix timestamp
                date = datetime.fromtimestamp(timestamp).date()

            # Prepare sample (defer tensorization until later for efficiency)
            sample = {
                'codebook': doc['codebook_index'],  # Field name is 'codebook_index' in database
                'features': doc['features'],  # Keep as list for now
                'timestamp': timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp,
                'target': doc['target'],
                'fold_id': doc.get('fold_id', 0)  # Test data may not have fold_id
            }

            episodes_by_date[date].append(sample)

        # Create Episode objects (day-level first)
        # Use split_id=-1 for test data (no specific split)
        test_split_id = -1
        day_episodes = []

        for date in sorted(episodes_by_date.keys()):
            samples = episodes_by_date[date]
            day_episodes.append(Episode(test_split_id, date, samples))

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
                        test_split_id, day_episode.date, chunk_samples,
                        parent_id=parent_id, chunk_id=chunk_idx
                    )
                    episodes.append(chunk_episode)
                parent_id += 1

        # Tensorize all episode samples if enabled (batch operation for efficiency)
        if self.pre_tensorize:
            episodes = self._tensorize_episodes(episodes)

        logger(f'  Loaded {len(episodes)} test episodes', "INFO")
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