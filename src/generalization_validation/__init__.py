"""
Generalization Validation Module

Validates VQ-VAE and Prior model quality through three experiments:
1. VQ-VAE Reconstruction Generalization
2. Prior Model Quality
3. End-to-End Synthetic Data Quality
"""

from .vqvae_reconstruction import VQVAEReconstructionValidator
from .prior_quality import PriorQualityValidator
from .end_to_end_quality import EndToEndValidator
from .metrics import (
    compute_mmd,
    compute_ks_tests,
    compute_correlation_distance,
    compute_transition_matrix,
    extract_ngrams
)
from .visualization import (
    plot_umap_comparison,
    plot_reconstruction_error,
    plot_code_frequency,
    plot_transition_matrix,
    plot_ngram_comparison
)
from .data_loader import (
    load_validation_samples,
    load_synthetic_samples,
    load_vqvae_model,
    organize_codes_into_sequences
)

__all__ = [
    'VQVAEReconstructionValidator',
    'PriorQualityValidator',
    'EndToEndValidator',
    'compute_mmd',
    'compute_ks_tests',
    'compute_correlation_distance',
    'compute_transition_matrix',
    'extract_ngrams',
    'plot_umap_comparison',
    'plot_reconstruction_error',
    'plot_code_frequency',
    'plot_transition_matrix',
    'plot_ngram_comparison',
    'load_validation_samples',
    'load_synthetic_samples',
    'load_vqvae_model',
    'organize_codes_into_sequences'
]
