"""
VQ-VAE Configuration

Defines hyperparameter grids and training configurations for LOB representation learning.
"""

# =================================================================================================
# Hyperparameter Grid for Search - Budget-Optimized Configuration (WORST CASE)
# =================================================================================================
# Optimized for $50 budget with $0.71/hour GPU (70 hours available)
# Configuration: 8 configs × 28 splits (all splits for statistical validity)
# Budget based on WORST CASE: all 25 epochs, no early stopping
# =================================================================================================

HYPERPARAM_GRID = {
    'B': [1001],                    # Number of LOB bins (fixed)
    'K': [128, 512],                # Codebook size: small vs large extremes
    'D': [64],                      # Embedding dimension: single best estimate
    'n_conv_layers': [2, 3],        # Architecture depth: shallow vs deep
    'beta': [0.25, 0.5],            # Commitment loss coefficient: low vs high
    'lr': [1e-3],                   # Learning rate: single best estimate (fast)
    'dropout': [0.2],               # Dropout rate for regularization
    'use_ema': [True],              # EMA codebook updates (improved stability)
    'ema_decay': [0.99],            # EMA decay rate (standard value)
    'recon_loss_type': ['wasserstein']  # Reconstruction loss: Wasserstein for probability distributions
}
# Total: 2 × 1 × 2 × 2 × 1 × 1 × 1 × 1 × 1 = 8 configurations (worst-case budget)
# Parameter range: ~800K (K=128,D=64,n=2) to ~2.1M (K=512,D=64,n=3)
# ALL 28 splits tested per configuration (full statistical validity)
#
# Budget calculation (WORST CASE - all 25 epochs, no early stopping):
# - 25 epochs × 40s = 1000s per split = 16.7 min
# - 28 splits × 16.7 min = 467 min per config = 7.8 hours
# - 8 configs × 7.8 hours = 62.4 hours
# - Cost: 62.4 × $0.71 = $44.30 (12% under budget for safety)
#
# Expected case (with early stopping ~7 epochs):
# - 7 epochs × 40s = 280s per split = 4.7 min
# - 28 splits × 4.7 min = 132 min per config = 2.2 hours
# - 8 configs × 2.2 hours = 17.6 hours
# - Cost: 17.6 × $0.71 = $12.50 (much cheaper if early stopping works!)

# =================================================================================================
# Training Configuration (shared across all configs)
# =================================================================================================
# Optimized for AWS g4dn.xlarge GPU (NVIDIA T4)
# Hour accumulation + large batches = 50-80× speedup vs laptop CPU
#
# Extended training capacity:
# - max_epochs: 100 (up from 25) - model still learning at epoch 40
# - patience: 10 (up from 3) - allows learning through plateaus
# - Learning rate scheduler: ReduceLROnPlateau with patience=5
#   * Reduces LR by 0.5× when validation loss plateaus for 5 epochs
#   * Minimum LR: 1e-6 (prevents too-small learning rates)
# =================================================================================================

TRAINING_CONFIG = {
    'max_epochs': 100,              # Increased to allow more learning (model still learning at epoch 40)
    'patience': 10,                 # Increased patience for early stopping (allows plateaus)
    'hours_per_accumulation': 100,  # Accumulate 100 hours before processing
    'mini_batch_size': 2048,        # Large batches for GPU (was 32)
    'grad_clip_norm': 1.0,          # Gradient clipping
    'weight_decay': 1e-5,           # L2 regularization
    'usage_penalty_weight': 0.1,    # Codebook diversity penalty
    # Learning rate scheduler config
    'lr_scheduler_type': 'plateau', # 'plateau' or 'cosine' or None
    'lr_scheduler_factor': 0.5,     # ReduceLROnPlateau: multiply LR by this on plateau
    'lr_scheduler_patience': 5,     # ReduceLROnPlateau: epochs to wait before reducing LR
    'lr_min': 1e-6                  # Minimum learning rate
}

# Performance impact:
# - Hour accumulation: 100 hours × ~120 samples = 12,000 samples per batch
# - Mini-batch size: 2048 samples processed in parallel on GPU
# - GPU utilization: 70-85% (vs 20% without optimization)
# - Time per epoch: 2-3 seconds (vs 20 minutes on laptop)

# =================================================================================================
# Model Architecture Configuration
# =================================================================================================

MODEL_CONFIG = {
    # Convolutional encoder architecture
    'conv_channels': [32, 64, 128, 256],  # Progressive channel expansion
    'kernel_sizes': [9, 7, 5, 3],          # Decreasing kernel sizes
    'strides': [2, 2, 2, 2],               # Strided convolutions for downsampling
    'paddings': [4, 3, 2, 1]               # Matching paddings for size preservation
}