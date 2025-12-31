"""Experiment 1: VQ-VAE Reconstruction Generalization."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.logging import logger
from .data_loader import load_validation_samples, load_vqvae_model, decode_codes_batch
from .metrics import compute_ks_tests, compute_correlation_distance, compute_cosine_similarity
from .visualization import plot_umap_comparison, plot_reconstruction_error


class VQVAEReconstructionValidator:
    """Validates VQ-VAE reconstruction quality on unseen validation data."""

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
        (self.output_dir / "experiment1_vqvae_reconstruction").mkdir(parents=True, exist_ok=True)

    def validate_split(self, split_id: int) -> Dict:
        """
        Run VQ-VAE reconstruction validation for one split.

        Args:
            split_id: Split identifier

        Returns:
            Dictionary with validation metrics
        """
        logger('', "INFO")
        logger(f'Validating VQ-VAE reconstruction for split {split_id}...', "INFO")

        # Load validation data
        original_vectors, codebook_indices, _, _ = load_validation_samples(
            self.mongo_uri, self.db_name, split_id
        )

        n_samples = len(original_vectors)
        logger(f'  Validation samples: {n_samples:,}', "INFO")

        # Log original data statistics for liquidity analysis
        logger('  Original data statistics:', "INFO")
        logger(f'    Min: {np.min(original_vectors):.6f}', "INFO")
        logger(f'    Max: {np.max(original_vectors):.6f}', "INFO")
        logger(f'    Mean: {np.mean(original_vectors):.6f}', "INFO")
        logger(f'    Std: {np.std(original_vectors):.6f}', "INFO")

        # Load VQ-VAE model
        model_path = self.vqvae_model_dir / f"split_{split_id}_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"VQ-VAE model not found: {model_path}")

        vqvae_model = load_vqvae_model(model_path, self.device)

        # Decode validation codes to get reconstructions
        logger('  Decoding validation codes...', "INFO")
        reconstructed_vectors = decode_codes_batch(
            vqvae_model, codebook_indices, self.device, batch_size=512
        )

        # Compute reconstruction metrics
        logger('  Computing reconstruction metrics...', "INFO")

        # Overall MSE
        mse_overall = np.mean((original_vectors - reconstructed_vectors) ** 2)

        # Per-feature MSE
        mse_per_feature = np.mean((original_vectors - reconstructed_vectors) ** 2, axis=0)

        # MAE
        mae_overall = np.mean(np.abs(original_vectors - reconstructed_vectors))

        # Find worst features
        worst_features = np.argsort(mse_per_feature)[-10:][::-1]

        logger(f'  Overall MSE: {mse_overall:.6f}', "INFO")
        logger(f'  Overall MAE: {mae_overall:.6f}', "INFO")
        logger(f'  Worst feature MSE: {mse_per_feature[worst_features[0]]:.6f} (feature {worst_features[0]})', "INFO")

        # KS tests
        logger('  Running KS tests...', "INFO")
        ks_results = compute_ks_tests(original_vectors, reconstructed_vectors)
        logger(f'  Mean KS statistic: {ks_results["mean_ks_statistic"]:.6f}', "INFO")
        logger(f'  Rejection rate: {ks_results["rejection_rate"]:.4f}', "INFO")

        # Correlation distance
        logger('  Computing correlation distance...', "INFO")
        corr_results = compute_correlation_distance(original_vectors, reconstructed_vectors)

        # Diagnostic logging for correlation matrices
        corr_orig = corr_results['corr_original']
        corr_recon = corr_results['corr_synthetic']

        # Check if correlation matrices are mostly diagonal
        n_features = corr_orig.shape[0]
        off_diagonal_orig = np.abs(corr_orig - np.eye(n_features))
        off_diagonal_recon = np.abs(corr_recon - np.eye(n_features))

        logger(f'  Original corr matrix off-diagonal mean: {off_diagonal_orig.mean():.6f}', "INFO")
        logger(f'  Original corr matrix off-diagonal max: {off_diagonal_orig.max():.6f}', "INFO")
        logger(f'  Reconstructed corr matrix off-diagonal mean: {off_diagonal_recon.mean():.6f}', "INFO")
        logger(f'  Reconstructed corr matrix off-diagonal max: {off_diagonal_recon.max():.6f}', "INFO")

        logger(f'  Correlation Frobenius correlation: {corr_results["frobenius_correlation"]:.6f}', "INFO")
        logger(f'  Mean absolute difference: {corr_results["mean_absolute_diff"]:.6f}', "INFO")

        # Cosine similarity
        logger('  Computing cosine similarity...', "INFO")
        cosine_results = compute_cosine_similarity(original_vectors, reconstructed_vectors)
        logger(f'  Mean cosine similarity: {cosine_results["mean_cosine_similarity"]:.6f}', "INFO")
        logger(f'  Min cosine similarity: {cosine_results["min_cosine_similarity"]:.6f}', "INFO")

        # Visualizations
        split_output_dir = self.output_dir / "experiment1_vqvae_reconstruction" / f"split_{split_id}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # UMAP visualization
        logger('  Generating UMAP visualization...', "INFO")
        plot_umap_comparison(
            original_vectors, reconstructed_vectors,
            title='Original vs. Reconstruction',
            save_path=split_output_dir / f"umap_reconstruction_split_{split_id}.png",
            method='umap',
            label_second='Reconstruction'
        )

        # Per-feature reconstruction error plot
        logger('  Plotting reconstruction errors...', "INFO")
        plot_reconstruction_error(
            mse_per_feature,
            save_path=split_output_dir / f"reconstruction_error_split_{split_id}.png"
        )

        # Compile results
        results = {
            'split_id': split_id,
            'n_samples': n_samples,
            'original_min': float(np.min(original_vectors)),
            'original_max': float(np.max(original_vectors)),
            'original_mean': float(np.mean(original_vectors)),
            'original_std': float(np.std(original_vectors)),
            'mse_overall': float(mse_overall),
            'mae_overall': float(mae_overall),
            'mse_per_feature_mean': float(np.mean(mse_per_feature)),
            'mse_per_feature_std': float(np.std(mse_per_feature)),
            'mse_per_feature_max': float(np.max(mse_per_feature)),
            'worst_features': worst_features.tolist(),
            'worst_feature_mses': mse_per_feature[worst_features].tolist(),
            'ks_mean_statistic': float(ks_results['mean_ks_statistic']),
            'ks_max_statistic': float(ks_results['max_ks_statistic']),
            'ks_rejection_rate': float(ks_results['rejection_rate']),
            'corr_frobenius_correlation': float(corr_results['frobenius_correlation']),
            'corr_mean_abs_diff': float(corr_results['mean_absolute_diff']),
            'corr_max_abs_diff': float(corr_results['max_absolute_diff']),
            'cosine_similarity_mean': float(cosine_results['mean_cosine_similarity']),
            'cosine_similarity_std': float(cosine_results['std_cosine_similarity']),
            'cosine_similarity_min': float(cosine_results['min_cosine_similarity']),
            'cosine_similarity_max': float(cosine_results['max_cosine_similarity']),
            'cosine_similarity_median': float(cosine_results['median_cosine_similarity'])
        }

        logger(f'  ✓ Split {split_id} validation complete', "INFO")

        return results
