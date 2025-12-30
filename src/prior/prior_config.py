"""
Prior Model Configuration

Hyperparameter grids and training configurations for latent prior learning.
"""

# =================================================================================================
# Hyperparameter Grid for Search
# =================================================================================================
# Optimized for receptive field efficiency and matching VQ-VAE embedding dimension
#
# Key improvements:
# - embedding_dim=64 matches VQ-VAE latent dimension
# - Repeating dilation patterns prevent excessive receptive fields (seq_len=120)
# - n_channels=[64, 80] restores capacity (learned from underfitting with n_channels=32)
# - n_layers=[10, 12, 16] focuses on depth for capacity
# - 6 focused configurations
#
# Expected runtime (assuming ~5s per epoch with prefetching):
# - Worst case (25 epochs per split): 6 configs × 28 splits × 25 epochs × 5s = 21,000s = 5.8 hours = $4.12
# - Expected case (early stopping ~10 epochs): 6 configs × 28 splits × 10 epochs × 5s = 8,400s = 2.3 hours = $1.63
# =================================================================================================

PRIOR_HYPERPARAM_GRID = {
    # Architecture parameters
    'embedding_dim': [64],               # Match VQ-VAE embedding dimension
    'n_layers': [10, 12, 16],            # Depth with repeating dilations (RF controlled)
    'n_channels': [64, 80],              # Restore capacity (64 or 80 channels)
    'kernel_size': [2],                  # Standard for causal convolutions

    # Training parameters
    'learning_rate': [1e-3],             # Adam learning rate
    'dropout': [0.15],                   # Regularization
}
# Total: 1 × 3 × 2 × 1 × 1 × 1 = 6 configurations
# Receptive field: Max 127 timesteps (fully covers seq_len=120)
# Parameter range: ~270K-710K (K=128) or ~287K-726K (K=512)

# =================================================================================================
# Training Configuration
# =================================================================================================

PRIOR_TRAINING_CONFIG = {
    'max_epochs': 25,                    # Aligned with VQVAE training
    'patience': 3,                       # Early stopping patience (aligned with VQVAE)
    'min_delta': 0.001,                  # Minimum improvement threshold
    'hours_per_accumulation': 100,      # Match VQ-VAE for GPU efficiency
    'sequence_batch_size': 32,           # Number of sequences per GPU batch
    'seq_len': 120,                      # 1 hour = ~120 LOB snapshots
    'grad_clip_norm': 1.0,
    'weight_decay': 1e-5,
}

# =================================================================================================
# Generation Configuration
# =================================================================================================

GENERATION_CONFIG = {
    'sequences_per_split': 100,          # Number of synthetic sequences per split
    'seq_len': 120,                      # 1 hour sequences
    'temperature': 1.0,                  # Sampling temperature (match training)
    'generation_batch_size': 32,         # Generate 32 sequences at once (GPU)
}