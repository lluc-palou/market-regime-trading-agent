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
        
        # Initialize model (with target prediction enabled)
        self.model = LatentPriorCNN(
            codebook_size=codebook_size,
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            n_channels=config['n_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            predict_target=True
        ).to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=PRIOR_TRAINING_CONFIG['weight_decay']
        )

        # Loss functions
        self.codebook_criterion = nn.CrossEntropyLoss()
        self.target_criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers

        # Multi-task loss weight (default 0.5 = equal weight)
        self.task_loss_weight = config.get('task_loss_weight', 0.5)
        
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
            train_loss, train_codebook_loss, train_target_loss = self._train_epoch(all_hours, epoch)

            # Validate
            val_loss, val_codebook_loss, val_target_loss = self._validate_epoch(all_hours, epoch)

            epoch_duration = time.time() - epoch_start

            # Log with both loss components
            logger(
                f'Epoch {epoch+1}/{PRIOR_TRAINING_CONFIG["max_epochs"]} '
                f'[{epoch_duration:.1f}s] - '
                f'train_loss: {train_loss:.4f} '
                f'(codebook: {train_codebook_loss:.4f}, target: {train_target_loss:.4f}), '
                f'val_loss: {val_loss:.4f} '
                f'(codebook: {val_codebook_loss:.4f}, target: {val_target_loss:.4f})',
                "INFO"
            )

            train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_codebook_loss': train_codebook_loss,
                'train_target_loss': train_target_loss,
                'val_loss': val_loss,
                'val_codebook_loss': val_codebook_loss,
                'val_target_loss': val_target_loss
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
            'final_train_codebook_loss': train_history[self.best_epoch]['train_codebook_loss'] if train_history else None,
            'final_train_target_loss': train_history[self.best_epoch]['train_target_loss'] if train_history else None,
            'final_val_codebook_loss': train_history[self.best_epoch]['val_codebook_loss'] if train_history else None,
            'final_val_target_loss': train_history[self.best_epoch]['val_target_loss'] if train_history else None,
            'epochs_trained': len(train_history),
            'train_history': train_history  # Include full history for detailed analysis
        }
    
    def _train_epoch(self, all_hours: List[datetime], epoch: int) -> tuple[float, float, float]:
        """Train for one epoch with hour accumulation and prefetching."""
        self.model.train()

        epoch_loss = 0.0
        epoch_codebook_loss = 0.0
        epoch_target_loss = 0.0
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
                return None, None

            code_sequences, target_sequences, _ = create_sequences_from_codes(
                latent_codes,
                PRIOR_TRAINING_CONFIG['seq_len']
            )

            return code_sequences, target_sequences

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
                result = future.result() if future else (None, None)
                code_sequences, target_sequences = result

                # Start loading NEXT batch in background (while GPU processes current)
                if i + 1 < len(hour_groups):
                    future = executor.submit(load_batch_func, hour_groups[i + 1])
                else:
                    future = None

                if code_sequences is None:
                    continue

                # Process on GPU while next batch loads in background
                num_sequences = code_sequences.size(0)
                batch_size = PRIOR_TRAINING_CONFIG['sequence_batch_size']

                for j in range(0, num_sequences, batch_size):
                    code_batch = code_sequences[j:j+batch_size].to(self.device)
                    target_batch = target_sequences[j:j+batch_size].to(self.device)

                    # Input: all except last timestep
                    code_input = code_batch[:, :-1]
                    target_input = target_batch[:, :-1]

                    # Target: all except first timestep (shifted for autoregressive)
                    code_target = code_batch[:, 1:]
                    target_target = target_batch[:, 1:]

                    # Forward
                    self.optimizer.zero_grad()
                    codebook_logits, target_preds = self.model(code_input, target_input)

                    # Multi-task loss
                    codebook_loss = self.codebook_criterion(
                        codebook_logits.reshape(-1, codebook_logits.size(-1)),
                        code_target.reshape(-1)
                    )
                    target_loss = self.target_criterion(
                        target_preds.reshape(-1),
                        target_target.reshape(-1)
                    )

                    # Weighted combination
                    loss = self.task_loss_weight * codebook_loss + (1 - self.task_loss_weight) * target_loss

                    # Backward
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        PRIOR_TRAINING_CONFIG['grad_clip_norm']
                    )

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_codebook_loss += codebook_loss.item()
                    epoch_target_loss += target_loss.item()
                    num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_codebook_loss = epoch_codebook_loss / max(num_batches, 1)
        avg_target_loss = epoch_target_loss / max(num_batches, 1)

        return avg_loss, avg_codebook_loss, avg_target_loss
    
    def _validate_epoch(self, all_hours: List[datetime], epoch: int) -> tuple[float, float, float]:
        """Validate for one epoch with prefetching."""
        self.model.eval()

        epoch_loss = 0.0
        epoch_codebook_loss = 0.0
        epoch_target_loss = 0.0
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
                return None, None

            code_sequences, target_sequences, _ = create_sequences_from_codes(
                latent_codes,
                PRIOR_TRAINING_CONFIG['seq_len']
            )

            return code_sequences, target_sequences

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
                    result = future.result() if future else (None, None)
                    code_sequences, target_sequences = result

                    # Start loading next batch
                    if i + 1 < len(hour_groups):
                        future = executor.submit(load_val_batch_func, hour_groups[i + 1])
                    else:
                        future = None

                    if code_sequences is None:
                        continue

                    # Process on GPU while next batch loads
                    num_sequences = code_sequences.size(0)
                    batch_size = PRIOR_TRAINING_CONFIG['sequence_batch_size']

                    for j in range(0, num_sequences, batch_size):
                        code_batch = code_sequences[j:j+batch_size].to(self.device)
                        target_batch = target_sequences[j:j+batch_size].to(self.device)

                        code_input = code_batch[:, :-1]
                        target_input = target_batch[:, :-1]

                        code_target = code_batch[:, 1:]
                        target_target = target_batch[:, 1:]

                        codebook_logits, target_preds = self.model(code_input, target_input)

                        # Multi-task loss
                        codebook_loss = self.codebook_criterion(
                            codebook_logits.reshape(-1, codebook_logits.size(-1)),
                            code_target.reshape(-1)
                        )
                        target_loss = self.target_criterion(
                            target_preds.reshape(-1),
                            target_target.reshape(-1)
                        )

                        loss = self.task_loss_weight * codebook_loss + (1 - self.task_loss_weight) * target_loss

                        epoch_loss += loss.item()
                        epoch_codebook_loss += codebook_loss.item()
                        epoch_target_loss += target_loss.item()
                        num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_codebook_loss = epoch_codebook_loss / max(num_batches, 1)
        avg_target_loss = epoch_target_loss / max(num_batches, 1)

        return avg_loss, avg_codebook_loss, avg_target_loss