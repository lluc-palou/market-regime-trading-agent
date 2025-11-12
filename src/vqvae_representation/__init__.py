"""
VQ-VAE Representation Learning Module

Provides functionality for learning discrete latent representations of LOB data using VQ-VAE:
- Hyperparameter search across CPCV splits (Phase 1)
- Production model training per split (Phase 2)
- Latent representation generation

Module structure follows the pattern from feature_standardization and feature_transformation.
"""

from .model import VQVAEModel, Encoder, Decoder, VectorQuantizer
from .trainer import VQVAETrainer
from .data_loader import (
    discover_splits,
    get_split_info,
    get_all_hours,
    load_hourly_batch,
    load_hourly_batch_dataframe
)
from .hyperparameter_search import run_hyperparameter_search
from .production_trainer import run_production_training
from .latent_generator import LatentGenerator
from .config import (
    HYPERPARAM_GRID,
    TRAINING_CONFIG,
    MODEL_CONFIG
)

__all__ = [
    # Model components
    'VQVAEModel',
    'Encoder',
    'Decoder',
    'VectorQuantizer',
    
    # Training
    'VQVAETrainer',
    
    # Data loading
    'discover_splits',
    'get_split_info',
    'get_all_hours',
    'load_hourly_batch',
    'load_hourly_batch_dataframe',
    
    # Hyperparameter search (Phase 1)
    'run_hyperparameter_search',
    
    # Production training (Phase 2)
    'run_production_training',
    'LatentGenerator',
    
    # Configuration
    'HYPERPARAM_GRID',
    'TRAINING_CONFIG',
    'MODEL_CONFIG'
]