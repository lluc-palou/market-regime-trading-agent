"""
Prior Model Module

Provides functionality for learning prior distributions over VQ-VAE latent codes
and generating synthetic LOB sequences.

Components:
- Prior model architecture (Causal CNN)
- Hyperparameter search
- Production training
- Synthetic data generation
"""

from .prior_model import LatentPriorCNN, CausalConv1d, ResidualBlock
from .prior_trainer import PriorTrainer
from .prior_data_loader import (
    load_latent_codes_for_hours,
    create_sequences_from_codes,
    load_sequences_for_split,
    get_sequence_dataset_size
)
from .prior_hyperparameter_search import run_prior_hyperparameter_search
from .synthetic_generator import SyntheticLOBGenerator, load_models_for_generation
from .prior_config import (
    PRIOR_HYPERPARAM_GRID,
    PRIOR_TRAINING_CONFIG,
    GENERATION_CONFIG
)

__all__ = [
    # Model components
    'LatentPriorCNN',
    'CausalConv1d',
    'ResidualBlock',
    
    # Training
    'PriorTrainer',
    
    # Data loading
    'load_latent_codes_for_hours',
    'create_sequences_from_codes',
    'load_sequences_for_split',
    'get_sequence_dataset_size',
    
    # Hyperparameter search
    'run_prior_hyperparameter_search',

    # Synthetic generation (production)
    'SyntheticLOBGenerator',
    'load_models_for_generation',
    
    # Configuration
    'PRIOR_HYPERPARAM_GRID',
    'PRIOR_TRAINING_CONFIG',
    'GENERATION_CONFIG'
]