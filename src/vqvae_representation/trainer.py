"""
VQ-VAE Trainer

Implements two-pass training following the pattern from feature_standardization/processor.py:

Pass 1 (Training): Stream through training hours, train model with early stopping
Pass 2 (Validation): Stream through validation hours, evaluate frozen model

Memory-efficient: Processes data hour-by-hour without full accumulation.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.utils.logging import logger
from .model import VQVAEModel
from .data_loader import get_all_hours, load_hourly_batch
from .config import TRAINING_CONFIG


class VQVAETrainer:
    """
    Trains VQ-VAE with two-pass approach on a single split.
    
    Similar to EWMAHalfLifeProcessor pattern:
    - Pass 1: Train on role='train' samples with early stopping
    - Pass 2: Validate on role='validation' samples
    """
    
    def __init__(
        self,
        spark,
        db_name: str,
        split_collection: str,
        device: torch.device,
        config: Dict
    ):
        """
        Initialize VQ-VAE trainer.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            split_collection: Split collection name (e.g., 'split_0_input')
            device: torch device (cuda or cpu)
            config: Hyperparameter configuration
        """
        self.spark = spark
        self.db_name = db_name
        self.split_collection = split_collection
        self.device = device
        self.config = config
        
        # Initialize model
        self.model = VQVAEModel(config).to(device)
        
        # Initialize optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
    
    def train_split(self, all_hours: List[datetime]) -> Dict:
        """
        Train and validate on a single split.
        
        Args:
            all_hours: List of hourly time windows in split
            
        Returns:
            Dictionary with training results:
                - best_val_loss: Best validation loss achieved
                - best_epoch: Epoch where best validation occurred
                - final_train_losses: Training losses at best epoch
                - final_val_losses: Validation losses at best epoch
                - epochs_trained: Total epochs trained
        """
        logger(f'Training VQ-VAE on {self.split_collection}', "INFO")
        logger(f'Hours available: {len(all_hours)}', "INFO")
        logger(f'Config: {self.config}', "INFO")
        
        # Pass 1: Training with early stopping
        train_history = self._pass1_train(all_hours)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Pass 2: Final validation on best model
        val_losses = self._pass2_validate(all_hours)
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_losses': train_history[self.best_epoch],
            'final_val_losses': val_losses,
            'epochs_trained': len(train_history)
        }
    
    def _pass1_train(self, all_hours: List[datetime]) -> List[Dict]:
        """
        Pass 1: Train model on training data with early stopping.
        
        For each epoch:
            - Stream through hours
            - Train on role='train' samples
            - Validate on role='validation' samples after each epoch
            - Check early stopping
        
        Args:
            all_hours: List of hourly time windows
            
        Returns:
            List of training metrics per epoch
        """
        logger('', "INFO")
        logger('Pass 1: Training phase', "INFO")
        logger('=' * 80, "INFO")
        
        train_history = []
        
        for epoch in range(TRAINING_CONFIG['max_epochs']):
            epoch_start = time.time()
            
            # Train for one epoch
            train_losses = self._train_epoch(all_hours, epoch)
            
            # Validate after each epoch
            val_losses = self._validate_epoch(all_hours, epoch)
            
            epoch_duration = time.time() - epoch_start
            
            # Log epoch results
            logger(
                f'Epoch {epoch+1}/{TRAINING_CONFIG["max_epochs"]} '
                f'[{epoch_duration:.1f}s] - '
                f'train_loss: {train_losses["total_loss"]:.4f}, '
                f'val_loss: {val_losses["total_loss"]:.4f}, '
                f'val_perplexity: {val_losses["perplexity"]:.2f}, '
                f'val_usage: {val_losses["codebook_usage"]:.3f}',
                "INFO"
            )
            
            # Save training history
            train_history.append({
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses
            })
            
            # Check for improvement
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                logger(f'  → New best validation loss: {self.best_val_loss:.4f}', "INFO")
            
            # Early stopping check
            if epoch - self.best_epoch >= TRAINING_CONFIG['patience']:
                logger(f'Early stopping at epoch {epoch+1} (patience={TRAINING_CONFIG["patience"]})', "INFO")
                break
        
        logger(f'Training complete. Best epoch: {self.best_epoch+1}, Best val loss: {self.best_val_loss:.4f}', "INFO")
        
        return train_history
    
    def _train_epoch(self, all_hours: List[datetime], epoch: int) -> Dict:
        """
        Train for one epoch with hour accumulation for GPU optimization.
        
        Instead of processing 1 hour at a time (120 samples), accumulates multiple
        hours (default: 100) to create larger batches that better utilize GPU.
        
        Args:
            all_hours: List of hourly time windows
            epoch: Current epoch number
            
        Returns:
            Dictionary with aggregated training losses
        """
        self.model.train()
        
        hours_per_acc = TRAINING_CONFIG['hours_per_accumulation']
        mini_batch_size = TRAINING_CONFIG['mini_batch_size']
        
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'commitment_loss': 0.0,
            'codebook_loss': 0.0,
            'usage_penalty': 0.0,
            'codebook_usage': 0.0,
            'perplexity': 0.0,
            'num_batches': 0
        }
        
        # Process hours in groups for better GPU utilization
        for hour_idx in range(0, len(all_hours), hours_per_acc):
            # Get hour group (e.g., 100 hours at once)
            hour_group = all_hours[hour_idx:min(hour_idx + hours_per_acc, len(all_hours))]
            
            # Accumulate all hours in this group
            accumulated_samples = []
            for hour in hour_group:
                hour_end = hour + timedelta(hours=1)
                
                # Load training batch for this hour
                batch = load_hourly_batch(
                    self.spark,
                    self.db_name,
                    self.split_collection,
                    hour,
                    hour_end,
                    role='train'
                )
                
                if batch is not None:
                    accumulated_samples.append(batch)
            
            if not accumulated_samples:
                continue
            
            # Combine all hours: e.g., 100 hours × 120 samples = 12,000 samples
            large_batch = torch.cat(accumulated_samples, dim=0)
            
            # Process in mini-batches for GPU efficiency
            num_samples = large_batch.size(0)
            
            for i in range(0, num_samples, mini_batch_size):
                mini_batch = large_batch[i:i+mini_batch_size].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                x_recon, loss_dict = self.model(mini_batch)
                
                # Compute total loss with regularization
                total_loss, loss_components = self._compute_total_loss(
                    loss_dict,
                    self.config['beta']
                )
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    TRAINING_CONFIG['grad_clip_norm']
                )
                self.optimizer.step()
                
                # Accumulate metrics
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['recon_loss'] += loss_components['recon_loss']
                epoch_metrics['commitment_loss'] += loss_components['commitment_loss']
                epoch_metrics['codebook_loss'] += loss_components['codebook_loss']
                epoch_metrics['usage_penalty'] += loss_components['usage_penalty']
                epoch_metrics['codebook_usage'] += loss_dict['codebook_usage']
                epoch_metrics['perplexity'] += loss_dict['perplexity']
                epoch_metrics['num_batches'] += 1
        
        # Average metrics
        if epoch_metrics['num_batches'] > 0:
            for key in epoch_metrics:
                if key != 'num_batches':
                    epoch_metrics[key] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    def _validate_epoch(self, all_hours: List[datetime], epoch: int) -> Dict:
        """
        Validate for one epoch with hour accumulation.
        
        Args:
            all_hours: List of hourly time windows
            epoch: Current epoch number
            
        Returns:
            Dictionary with aggregated validation losses
        """
        self.model.eval()
        
        hours_per_acc = TRAINING_CONFIG['hours_per_accumulation']
        mini_batch_size = TRAINING_CONFIG['mini_batch_size']
        
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'commitment_loss': 0.0,
            'codebook_loss': 0.0,
            'usage_penalty': 0.0,
            'codebook_usage': 0.0,
            'perplexity': 0.0,
            'num_batches': 0
        }
        
        with torch.no_grad():
            # Process hours in groups
            for hour_idx in range(0, len(all_hours), hours_per_acc):
                hour_group = all_hours[hour_idx:min(hour_idx + hours_per_acc, len(all_hours))]
                
                # Accumulate validation samples
                accumulated_samples = []
                for hour in hour_group:
                    hour_end = hour + timedelta(hours=1)
                    
                    # Load validation batch
                    batch = load_hourly_batch(
                        self.spark,
                        self.db_name,
                        self.split_collection,
                        hour,
                        hour_end,
                        role='validation'
                    )
                    
                    if batch is not None:
                        accumulated_samples.append(batch)
                
                if not accumulated_samples:
                    continue
                
                # Combine all hours
                large_batch = torch.cat(accumulated_samples, dim=0)
                
                # Process in mini-batches
                num_samples = large_batch.size(0)
                
                for i in range(0, num_samples, mini_batch_size):
                    mini_batch = large_batch[i:i+mini_batch_size].to(self.device)
                    
                    # Forward pass
                    x_recon, loss_dict = self.model(mini_batch)
                    
                    # Compute total loss
                    total_loss, loss_components = self._compute_total_loss(
                        loss_dict,
                        self.config['beta']
                    )
                    
                    # Accumulate metrics
                    epoch_metrics['total_loss'] += total_loss.item()
                    epoch_metrics['recon_loss'] += loss_components['recon_loss']
                    epoch_metrics['commitment_loss'] += loss_components['commitment_loss']
                    epoch_metrics['codebook_loss'] += loss_components['codebook_loss']
                    epoch_metrics['usage_penalty'] += loss_components['usage_penalty']
                    epoch_metrics['codebook_usage'] += loss_dict['codebook_usage']
                    epoch_metrics['perplexity'] += loss_dict['perplexity']
                    epoch_metrics['num_batches'] += 1
        
        # Average metrics
        if epoch_metrics['num_batches'] > 0:
            for key in epoch_metrics:
                if key != 'num_batches':
                    epoch_metrics[key] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    def _pass2_validate(self, all_hours: List[datetime]) -> Dict:
        """
        Pass 2: Final validation on frozen best model with hour accumulation.
        
        Streams through validation hours and computes final metrics.
        
        Args:
            all_hours: List of hourly time windows
            
        Returns:
            Dictionary with final validation metrics
        """
        logger('', "INFO")
        logger('Pass 2: Final validation on best model', "INFO")
        logger('=' * 80, "INFO")
        
        self.model.eval()
        
        hours_per_acc = TRAINING_CONFIG['hours_per_accumulation']
        mini_batch_size = TRAINING_CONFIG['mini_batch_size']
        
        final_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'commitment_loss': 0.0,
            'codebook_loss': 0.0,
            'usage_penalty': 0.0,
            'codebook_usage': 0.0,
            'perplexity': 0.0,
            'num_batches': 0,
            'num_samples': 0
        }
        
        with torch.no_grad():
            # Process hours in groups
            for hour_idx in range(0, len(all_hours), hours_per_acc):
                hour_group = all_hours[hour_idx:min(hour_idx + hours_per_acc, len(all_hours))]
                
                # Accumulate validation samples
                accumulated_samples = []
                for hour in hour_group:
                    hour_end = hour + timedelta(hours=1)
                    
                    # Load validation batch
                    batch = load_hourly_batch(
                        self.spark,
                        self.db_name,
                        self.split_collection,
                        hour,
                        hour_end,
                        role='validation'
                    )
                    
                    if batch is not None:
                        accumulated_samples.append(batch)
                
                if not accumulated_samples:
                    continue
                
                # Combine all hours
                large_batch = torch.cat(accumulated_samples, dim=0)
                final_metrics['num_samples'] += large_batch.size(0)
                
                # Process in mini-batches
                for i in range(0, large_batch.size(0), mini_batch_size):
                    mini_batch = large_batch[i:i+mini_batch_size].to(self.device)
                    
                    # Forward pass
                    x_recon, loss_dict = self.model(mini_batch)
                    
                    # Compute total loss
                    total_loss, loss_components = self._compute_total_loss(
                        loss_dict,
                        self.config['beta']
                    )
                    
                    # Accumulate metrics
                    final_metrics['total_loss'] += total_loss.item()
                    final_metrics['recon_loss'] += loss_components['recon_loss']
                    final_metrics['commitment_loss'] += loss_components['commitment_loss']
                    final_metrics['codebook_loss'] += loss_components['codebook_loss']
                    final_metrics['usage_penalty'] += loss_components['usage_penalty']
                    final_metrics['codebook_usage'] += loss_dict['codebook_usage']
                    final_metrics['perplexity'] += loss_dict['perplexity']
                    final_metrics['num_batches'] += 1
        
        # Average metrics
        if final_metrics['num_batches'] > 0:
            for key in final_metrics:
                if key not in ['num_batches', 'num_samples']:
                    final_metrics[key] /= final_metrics['num_batches']
        
        logger(
            f'Final validation - '
            f'loss: {final_metrics["total_loss"]:.4f}, '
            f'perplexity: {final_metrics["perplexity"]:.2f}, '
            f'usage: {final_metrics["codebook_usage"]:.3f}, '
            f'samples: {final_metrics["num_samples"]:,}',
            "INFO"
        )
        
        return final_metrics
    
    def _compute_total_loss(
        self,
        loss_dict: Dict,
        beta: float
    ) -> tuple:
        """
        Compute total loss with regularization.
        
        Components:
        - Reconstruction loss (MSE)
        - Commitment loss (weighted by beta)
        - Codebook loss
        - Usage penalty (encourage codebook diversity)
        
        Args:
            loss_dict: Dictionary from model forward pass
            beta: Commitment loss weight
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual components
        """
        recon_loss = loss_dict['recon_loss']
        commitment_loss = loss_dict['commitment_loss']
        codebook_loss = loss_dict['codebook_loss']
        codebook_usage = loss_dict['codebook_usage']
        
        # Usage penalty: penalize low codebook usage (encourage diversity)
        usage_penalty = (1.0 - codebook_usage) * TRAINING_CONFIG['usage_penalty_weight']
        
        # Total loss
        total_loss = recon_loss + beta * commitment_loss + codebook_loss + usage_penalty
        
        loss_components = {
            'recon_loss': recon_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'usage_penalty': usage_penalty
        }
        
        return total_loss, loss_components