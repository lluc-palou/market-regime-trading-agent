"""Experiment 3: End-to-End Synthetic Data Quality."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.logging import logger
from .data_loader import (
    load_validation_samples,
    load_synthetic_samples,
    load_vqvae_model,
    decode_codes_batch
)
from .metrics import compute_mmd, compute_ks_tests, compute_correlation_distance
from .visualization import plot_umap_comparison, plot_correlation_matrices


class EndToEndValidator:
    """Validates end-to-end pipeline quality using pre-generated synthetic data."""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        vqvae_model_dir: Path,
        output_dir: Path,
        device: torch.device
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.vqvae_model_dir = vqvae_model_dir
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        (self.output_dir / "experiment3_end_to_end").mkdir(parents=True, exist_ok=True)

    def validate_split(self, split_id: int) -> Dict:
        """
        Run end-to-end validation for one split using pre-generated synthetic data.

        Args:
            split_id: Split identifier

        Returns:
            Dictionary with validation metrics
        """
        logger('', "INFO")
        logger(f'Validating end-to-end quality for split {split_id}...', "INFO")

        # Load validation data
        original_vectors, codebook_indices, _ = load_validation_samples(
            self.mongo_uri, self.db_name, split_id
        )

        n_val_samples = len(original_vectors)
        logger(f'  Validation samples: {n_val_samples:,}', "INFO")

        # Load pre-generated synthetic data
        logger('  Loading synthetic data...', "INFO")
        synthetic_vectors, syn_codebook_indices, _ = load_synthetic_samples(
            self.mongo_uri, self.db_name, split_id
        )

        n_syn_samples = len(synthetic_vectors)
        logger(f'  Synthetic samples: {n_syn_samples:,}', "INFO")

        # Compute metrics comparing original validation data with synthetic data
        logger('  Computing MMD...', "INFO")
        mmd = compute_mmd(original_vectors, synthetic_vectors, kernel='rbf')
        logger(f'  MMD: {mmd:.6f}', "INFO")

        logger('  Running KS tests...', "INFO")
        ks_results = compute_ks_tests(original_vectors, synthetic_vectors)
        logger(f'  Mean KS statistic: {ks_results["mean_ks_statistic"]:.6f}', "INFO")
        logger(f'  Rejection rate: {ks_results["rejection_rate"]:.4f}', "INFO")

        logger('  Computing correlation distance...', "INFO")
        corr_results = compute_correlation_distance(original_vectors, synthetic_vectors)

        # Diagnostic logging for correlation matrices
        corr_orig = corr_results['corr_original']
        corr_syn = corr_results['corr_synthetic']

        # Check if correlation matrices are mostly diagonal
        n_features = corr_orig.shape[0]
        off_diagonal_orig = np.abs(corr_orig - np.eye(n_features))
        off_diagonal_syn = np.abs(corr_syn - np.eye(n_features))

        logger(f'  Original corr matrix off-diagonal mean: {off_diagonal_orig.mean():.6f}', "INFO")
        logger(f'  Original corr matrix off-diagonal max: {off_diagonal_orig.max():.6f}', "INFO")
        logger(f'  Synthetic corr matrix off-diagonal mean: {off_diagonal_syn.mean():.6f}', "INFO")
        logger(f'  Synthetic corr matrix off-diagonal max: {off_diagonal_syn.max():.6f}', "INFO")

        logger(f'  Correlation Frobenius correlation: {corr_results["frobenius_correlation"]:.6f}', "INFO")
        logger(f'  Mean absolute difference: {corr_results["mean_absolute_diff"]:.6f}', "INFO")

        # Visualizations
        split_output_dir = self.output_dir / "experiment3_end_to_end" / f"split_{split_id}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # UMAP visualization
        logger('  Generating UMAP visualization...', "INFO")
        plot_umap_comparison(
            original_vectors, synthetic_vectors,
            title='End-to-End Synthetic Generation',
            save_path=split_output_dir / f"umap_end_to_end_split_{split_id}.png",
            method='umap'
        )

        # Correlation matrices
        logger('  Plotting correlation matrices...', "INFO")
        plot_correlation_matrices(
            corr_results['corr_original'],
            corr_results['corr_synthetic'],
            save_path=split_output_dir / f"correlation_matrices_split_{split_id}.png",
            sample_features=100
        )

        # Marginal distribution comparisons (sample dimensions)
        logger('  Plotting marginal distributions...', "INFO")
        self._plot_marginals(
            original_vectors, synthetic_vectors,
            save_path=split_output_dir / f"marginals_split_{split_id}.png"
        )

        # Compile results
        results = {
            'split_id': split_id,
            'n_val_samples': n_val_samples,
            'n_syn_samples': n_syn_samples,
            'mmd': float(mmd),
            'ks_mean_statistic': float(ks_results['mean_ks_statistic']),
            'ks_max_statistic': float(ks_results['max_ks_statistic']),
            'ks_rejection_rate': float(ks_results['rejection_rate']),
            'corr_frobenius_correlation': float(corr_results['frobenius_correlation']),
            'corr_mean_abs_diff': float(corr_results['mean_absolute_diff']),
            'corr_max_abs_diff': float(corr_results['max_absolute_diff'])
        }

        logger(f'  ✓ Split {split_id} validation complete', "INFO")

        return results

    def _plot_marginals(
        self,
        val_data: np.ndarray,
        syn_data: np.ndarray,
        save_path: Path,
        n_features: int = 9
    ):
        """
        Plot marginal distributions for sample features.

        Args:
            val_data: Validation data
            syn_data: Synthetic data
            save_path: Path to save figure
            n_features: Number of features to plot
        """
        import matplotlib.pyplot as plt

        # Sample features evenly
        feature_indices = np.linspace(0, val_data.shape[1] - 1, n_features, dtype=int)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, feat_idx in enumerate(feature_indices):
            ax = axes[idx]

            # Get data for this feature
            val_feature = val_data[:, feat_idx]
            syn_feature = syn_data[:, feat_idx]

            # Compute data range for proper x-axis scaling
            data_min = min(val_feature.min(), syn_feature.min())
            data_max = max(val_feature.max(), syn_feature.max())
            data_range = data_max - data_min

            # Add 5% padding to the range for better visualization
            padding = 0.05 * data_range if data_range > 0 else 0.1
            xlim_min = data_min - padding
            xlim_max = data_max + padding

            # Create histograms and get the bin heights for y-axis scaling
            n_val, bins_val, _ = ax.hist(val_feature, bins=50, alpha=0.5, label='Validation',
                   density=True, color='#0072B2')
            n_syn, bins_syn, _ = ax.hist(syn_feature, bins=50, alpha=0.5, label='Synthetic',
                   density=True, color='#D55E00')

            # Set x-axis limits based on actual data range
            ax.set_xlim(xlim_min, xlim_max)

            # Set y-axis limits based on maximum density value with padding
            max_density = max(n_val.max(), n_syn.max())
            ax.set_ylim(0, max_density * 1.1)  # Add 10% padding at top

            ax.set_xlabel('Value', color='black', fontweight='bold')
            ax.set_ylabel('Density', color='black', fontweight='bold')
            ax.set_title(f'Feature {feat_idx}', color='black', fontweight='bold', pad=15)
            ax.legend()
            ax.tick_params(colors='black')
            for spine in ax.spines.values():
                spine.set_color('black')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
