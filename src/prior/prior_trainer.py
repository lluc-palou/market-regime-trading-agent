"""
Prior Trainer

Trains prior models on latent code sequences.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from src.utils.logging import logger
from .prior_model import LatentPriorCNN
from .prior_data_loader import load_latent_codes_for_hours, create_sequences_from_codes, get_sequence_dataset_size
from .prior_config import PRIOR_TRAINING_CONFIG


class PriorTrainer:
    """
    Trains prior model on latent code sequences.
    """
    
    def __init__(
        self,
        spark,
        db_name: str,
        split_collection: str,
        device: torch.device,
        config: Dict,
        codebook_size: int
    ):
        """
        Initialize prior trainer.
        
        Args:
            spark: SparkSession
            db_name: Database name
            split_collection: Split collection name
            device: torch device
            config: Hyperparameter configuration
            codebook_size: VQ-VAE codebook size (K)
        """
        self.spark = spark
        self.db_name = db_name
        self.split_collection = split_collection
        self.device = device
        self.config = config
        
        # Initialize model
        self.model = LatentPriorCNN(
            codebook_size=codebook_size,
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            n_channels=config['n_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=PRIOR_TRAINING_CONFIG['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
    
    def train_split(self, all_hours: List[datetime]) -> Dict:
        """
        Train prior model on a split.
        
        Args:
            all_hours: List of hourly time windows
            
        Returns:
            Training results dictionary
        """
        logger(f'Training Prior on {self.split_collection}', "INFO")
        logger(f'Hours available: {len(all_hours)}', "INFO")
        logger(f'Config: {self.config}', "INFO")
        logger(f'Receptive field: {self.model.compute_receptive_field()}', "INFO")
        
        # Get dataset sizes
        train_seq_count = get_sequence_dataset_size(
            self.spark, self.db_name, self.split_collection,
            'train', PRIOR_TRAINING_CONFIG['seq_len']
        )
        val_seq_count = get_sequence_dataset_size(
            self.spark, self.db_name, self.split_collection,
            'validation', PRIOR_TRAINING_CONFIG['seq_len']
        )
        
        logger(f'Estimated sequences - Train: {train_seq_count}, Val: {val_seq_count}', "INFO")
        
        if train_seq_count < 10:
            logger('Insufficient training sequences, skipping', "WARNING")
            return None
        
        # Training loop
        train_history = []
        
        for epoch in range(PRIOR_TRAINING_CONFIG['max_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self._train_epoch(all_hours, epoch)
            
            # Validate
            val_loss = self._validate_epoch(all_hours, epoch)
            
            epoch_duration = time.time() - epoch_start
            
            # Log
            logger(
                f'Epoch {epoch+1}/{PRIOR_TRAINING_CONFIG["max_epochs"]} '
                f'[{epoch_duration:.1f}s] - '
                f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}',
                "INFO"
            )
            
            train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Check for improvement
            if val_loss < self.best_val_loss - PRIOR_TRAINING_CONFIG['min_delta']:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                # Deep copy model state (clone tensors, not just dict structure)
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                logger(f'  → New best validation loss: {self.best_val_loss:.4f}', "INFO")
            
            # Early stopping
            if epoch - self.best_epoch >= PRIOR_TRAINING_CONFIG['patience']:
                logger(f'Early stopping at epoch {epoch+1}', "INFO")
                break
        
        logger(f'Training complete. Best epoch: {self.best_epoch+1}, Best val loss: {self.best_val_loss:.4f}', "INFO")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': train_history[self.best_epoch]['train_loss'] if train_history else None,
            'epochs_trained': len(train_history)
        }
    
    def _train_epoch(self, all_hours: List[datetime], epoch: int) -> float:
        """Train for one epoch with hour accumulation and prefetching."""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        hours_per_acc = PRIOR_TRAINING_CONFIG['hours_per_accumulation']

        # Helper function to load sequences for a hour group (for background thread)
        def load_batch_func(hour_group):
            hour_start = hour_group[0]
            hour_end = hour_group[-1] + timedelta(hours=1)

            latent_codes = load_latent_codes_for_hours(
                self.spark, self.db_name, self.split_collection,
                hour_start, hour_end, 'train'
            )

            if not latent_codes:
                return None

            sequences, _ = create_sequences_from_codes(
                latent_codes,
                PRIOR_TRAINING_CONFIG['seq_len']
            )

            return sequences

        # Create hour groups
        hour_groups = []
        for hour_idx in range(0, len(all_hours), hours_per_acc):
            hour_group = all_hours[hour_idx:min(hour_idx + hours_per_acc, len(all_hours))]
            hour_groups.append(hour_group)

        # Process with prefetching: load batch N+1 while processing batch N on GPU
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start loading first batch
            if hour_groups:
                future = executor.submit(load_batch_func, hour_groups[0])

            for i, hour_group in enumerate(hour_groups):
                # Wait for current batch to finish loading
                sequences = future.result() if future else None

                # Start loading NEXT batch in background (while GPU processes current)
                if i + 1 < len(hour_groups):
                    future = executor.submit(load_batch_func, hour_groups[i + 1])
                else:
                    future = None

                if sequences is None:
                    continue

                # Process on GPU while next batch loads in background
                num_sequences = sequences.size(0)
                batch_size = PRIOR_TRAINING_CONFIG['sequence_batch_size']

                for j in range(0, num_sequences, batch_size):
                    batch = sequences[j:j+batch_size].to(self.device)

                    # Input: all codes except last
                    input_seq = batch[:, :-1]
                    # Target: all codes except first
                    target_seq = batch[:, 1:]

                    # Forward
                    self.optimizer.zero_grad()
                    logits = self.model(input_seq)  # (batch, seq_len-1, codebook_size)

                    # Compute loss
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),  # (batch*(seq_len-1), codebook_size)
                        target_seq.reshape(-1)                 # (batch*(seq_len-1),)
                    )

                    # Backward
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        PRIOR_TRAINING_CONFIG['grad_clip_norm']
                    )

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

        return epoch_loss / max(num_batches, 1)
    
    def _validate_epoch(self, all_hours: List[datetime], epoch: int) -> float:
        """Validate for one epoch with prefetching."""
        self.model.eval()

        epoch_loss = 0.0
        num_batches = 0

        hours_per_acc = PRIOR_TRAINING_CONFIG['hours_per_accumulation']

        # Helper function for loading validation sequences
        def load_val_batch_func(hour_group):
            hour_start = hour_group[0]
            hour_end = hour_group[-1] + timedelta(hours=1)

            latent_codes = load_latent_codes_for_hours(
                self.spark, self.db_name, self.split_collection,
                hour_start, hour_end, 'validation'
            )

            if not latent_codes:
                return None

            sequences, _ = create_sequences_from_codes(
                latent_codes,
                PRIOR_TRAINING_CONFIG['seq_len']
            )

            return sequences

        # Create hour groups
        hour_groups = []
        for hour_idx in range(0, len(all_hours), hours_per_acc):
            hour_group = all_hours[hour_idx:min(hour_idx + hours_per_acc, len(all_hours))]
            hour_groups.append(hour_group)

        with torch.no_grad():
            # Process with prefetching
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Start loading first batch
                if hour_groups:
                    future = executor.submit(load_val_batch_func, hour_groups[0])

                for i, hour_group in enumerate(hour_groups):
                    # Wait for current batch
                    sequences = future.result() if future else None

                    # Start loading next batch
                    if i + 1 < len(hour_groups):
                        future = executor.submit(load_val_batch_func, hour_groups[i + 1])
                    else:
                        future = None

                    if sequences is None:
                        continue

                    # Process on GPU while next batch loads
                    num_sequences = sequences.size(0)
                    batch_size = PRIOR_TRAINING_CONFIG['sequence_batch_size']

                    for j in range(0, num_sequences, batch_size):
                        batch = sequences[j:j+batch_size].to(self.device)

                        input_seq = batch[:, :-1]
                        target_seq = batch[:, 1:]

                        logits = self.model(input_seq)

                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)),
                            target_seq.reshape(-1)
                        )

                        epoch_loss += loss.item()
                        num_batches += 1

        return epoch_loss / max(num_batches, 1)