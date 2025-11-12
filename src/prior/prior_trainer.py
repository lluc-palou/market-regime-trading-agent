"""
Prior Trainer

Trains prior models on latent code sequences.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, List

from src.utils.logging import logger
from .prior_model import LatentPriorCNN
from .prior_data_loader import load_sequences_for_split, get_sequence_dataset_size
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
                self.best_model_state = self.model.state_dict().copy()
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
        """Train for one epoch with hour accumulation."""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Load sequences with hour accumulation
        sequence_batches = load_sequences_for_split(
            self.spark,
            self.db_name,
            self.split_collection,
            all_hours,
            role='train',
            seq_len=PRIOR_TRAINING_CONFIG['seq_len'],
            hours_per_accumulation=PRIOR_TRAINING_CONFIG['hours_per_accumulation']
        )
        
        for sequences in sequence_batches:
            # sequences: (num_sequences, seq_len)
            num_sequences = sequences.size(0)
            
            # Process in mini-batches
            batch_size = PRIOR_TRAINING_CONFIG['sequence_batch_size']
            
            for i in range(0, num_sequences, batch_size):
                batch = sequences[i:i+batch_size].to(self.device)
                
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
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # Load validation sequences
            sequence_batches = load_sequences_for_split(
                self.spark,
                self.db_name,
                self.split_collection,
                all_hours,
                role='validation',
                seq_len=PRIOR_TRAINING_CONFIG['seq_len'],
                hours_per_accumulation=PRIOR_TRAINING_CONFIG['hours_per_accumulation']
            )
            
            for sequences in sequence_batches:
                num_sequences = sequences.size(0)
                batch_size = PRIOR_TRAINING_CONFIG['sequence_batch_size']
                
                for i in range(0, num_sequences, batch_size):
                    batch = sequences[i:i+batch_size].to(self.device)
                    
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