"""
Prior Model Configuration

Hyperparameter grids and training configurations for latent prior learning.
"""

# =================================================================================================
# Hyperparameter Grid for Search
# =================================================================================================
# Budget-optimized: 12 configurations, ~$43 total cost
# =================================================================================================

PRIOR_HYPERPARAM_GRID = {
    # Architecture parameters
    'embedding_dim': [128, 256],         # Code embedding size
    'n_layers': [10, 12],                # Causal CNN depth (receptive field)
    'n_channels': [96, 128, 160],        # Causal CNN width (capacity)
    'kernel_size': [2],                  # Standard for causal convolutions
    
    # Training parameters
    'learning_rate': [1e-3],             # Adam learning rate
    'dropout': [0.15],                   # Regularization
}
# Total: 2 × 2 × 3 = 12 configurations
# Parameter range: ~100K to ~270K (comparable to VQ-VAE decoder)

# =================================================================================================
# Training Configuration
# =================================================================================================

PRIOR_TRAINING_CONFIG = {
    'max_epochs': 50,
    'patience': 3,                       # Early stopping patience (aggressive)
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