"""
Generalization Validation Script (Stage 19)

Validates VQ-VAE and Prior model generalization through three experiments:
1. VQ-VAE Reconstruction Generalization
2. Prior Model Quality
3. End-to-End Synthetic Data Quality

This is Stage 19 in the pipeline - follows synthetic generation (Stage 17).

Input: - split_X_input collections with validation samples
       - VQ-VAE production models from Stage 14
       - Prior production models from Stage 16

Output: Validation metrics and visualizations saved to artifacts/generalization_validation/
        MLflow tracking with aggregate statistics

Usage:
    python scripts/19_generalization_validation.py
"""

import os
import sys
from pathlib import Path
import time
import json

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# Unicode/MLflow Fix for Windows - MUST BE FIRST!
# =================================================================================================
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
    os.environ['PYTHONUTF8'] = '1'

    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

# Patch MLflow emoji issue
try:
    from mlflow.tracking._tracking_service import client as mlflow_client

    _original_log_url = mlflow_client.TrackingServiceClient._log_url

    def _patched_log_url(self, run_id):
        try:
            run = self.get_run(run_id)
            run_name = run.info.run_name or run_id
            run_url = self._get_run_url(run.info.experiment_id, run_id)
            sys.stdout.write(f"[RUN] View run {run_name} at: {run_url}\n")
            sys.stdout.flush()
        except:
            pass

    mlflow_client.TrackingServiceClient._log_url = _patched_log_url
except:
    pass
# =================================================================================================

import torch
import mlflow
import numpy as np
from datetime import datetime

from src.utils.logging import logger
from src.generalization_validation import (
    VQVAEReconstructionValidator,
    PriorQualityValidator,
    EndToEndValidator
)

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_SUFFIX = "_input"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Generalization_Validation"

MONGO_URI = "mongodb://127.0.0.1:27017/"

# Model directories
VQVAE_MODEL_DIR = Path(REPO_ROOT) / "artifacts" / "vqvae_models" / "production"
PRIOR_MODEL_DIR = Path(REPO_ROOT) / "artifacts" / "prior_models" / "production"

# Output directory
OUTPUT_DIR = Path(REPO_ROOT) / "artifacts" / "generalization_validation"

# Prior sequence length
SEQ_LEN = 120

# Which experiments to run
RUN_EXPERIMENT_1 = True  # VQ-VAE Reconstruction
RUN_EXPERIMENT_2 = True  # Prior Quality
RUN_EXPERIMENT_3 = True  # End-to-End

# Splits to validate (None = all available)
SPLITS_TO_VALIDATE = None  # Or list like [0, 1, 2, 3]

# =================================================================================================
# Helper Functions
# =================================================================================================

def discover_splits() -> list:
    """Discover available splits from VQ-VAE models."""
    splits = []

    for model_file in VQVAE_MODEL_DIR.glob("split_*_model.pth"):
        # Extract split ID from filename
        split_id = int(model_file.stem.split('_')[1])
        splits.append(split_id)

    return sorted(splits)


def compute_aggregate_statistics(results: list, metric_name: str) -> dict:
    """
    Compute aggregate statistics across splits.

    Args:
        results: List of result dictionaries
        metric_name: Name of metric to aggregate

    Returns:
        Dictionary with mean, std, min, max
    """
    values = [r[metric_name] for r in results if metric_name in r]

    if len(values) == 0:
        return {}

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 100, "INFO")
    logger('GENERALIZATION VALIDATION (STAGE 19)', "INFO")
    logger('=' * 100, "INFO")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger('', "INFO")
    logger(f'Device: {device}', "INFO")

    if device.type == 'cuda':
        logger(f'CUDA Device: {torch.cuda.get_device_name(0)}', "INFO")
        logger(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB', "INFO")

    # Discover splits
    logger('', "INFO")
    logger('Discovering available splits...', "INFO")
    all_splits = discover_splits()

    if SPLITS_TO_VALIDATE is not None:
        splits_to_run = [s for s in SPLITS_TO_VALIDATE if s in all_splits]
    else:
        splits_to_run = all_splits

    logger(f'Found {len(all_splits)} splits: {all_splits}', "INFO")
    logger(f'Validating {len(splits_to_run)} splits: {splits_to_run}', "INFO")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    logger('', "INFO")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}', "INFO")
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")

    # Initialize validators
    validators = {}

    if RUN_EXPERIMENT_1:
        validators['exp1'] = VQVAEReconstructionValidator(
            MONGO_URI, DB_NAME, VQVAE_MODEL_DIR, OUTPUT_DIR, device
        )

    if RUN_EXPERIMENT_2:
        validators['exp2'] = PriorQualityValidator(
            MONGO_URI, DB_NAME, OUTPUT_DIR, device, seq_len=SEQ_LEN
        )

    if RUN_EXPERIMENT_3:
        validators['exp3'] = EndToEndValidator(
            MONGO_URI, DB_NAME, VQVAE_MODEL_DIR, OUTPUT_DIR, device
        )

    # Storage for results
    all_results = {
        'exp1': [],
        'exp2': [],
        'exp3': []
    }

    # Main MLflow run
    with mlflow.start_run(run_name=f"generalization_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("db_name", DB_NAME)
        mlflow.log_param("n_splits", len(splits_to_run))
        mlflow.log_param("run_exp1", RUN_EXPERIMENT_1)
        mlflow.log_param("run_exp2", RUN_EXPERIMENT_2)
        mlflow.log_param("run_exp3", RUN_EXPERIMENT_3)
        mlflow.log_param("seq_len", SEQ_LEN)

        # Process each split
        for split_idx, split_id in enumerate(splits_to_run):
            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger(f'SPLIT {split_id} ({split_idx + 1}/{len(splits_to_run)})', "INFO")
            logger('=' * 100, "INFO")

            split_start = time.time()

            with mlflow.start_run(run_name=f"split_{split_id}", nested=True):
                mlflow.log_param("split_id", split_id)

                # Experiment 1: VQ-VAE Reconstruction
                if RUN_EXPERIMENT_1:
                    try:
                        logger('', "INFO")
                        logger('EXPERIMENT 1: VQ-VAE RECONSTRUCTION GENERALIZATION', "INFO")
                        logger('-' * 100, "INFO")

                        exp1_results = validators['exp1'].validate_split(split_id)
                        all_results['exp1'].append(exp1_results)

                        # Log to MLflow
                        mlflow.log_metrics({
                            'exp1_mse_overall': exp1_results['mse_overall'],
                            'exp1_ks_rejection_rate': exp1_results['ks_rejection_rate'],
                            'exp1_corr_frobenius_correlation': exp1_results['corr_frobenius_correlation'],
                            'exp1_cosine_similarity_mean': exp1_results['cosine_similarity_mean']
                        })

                    except Exception as e:
                        logger(f'ERROR in Experiment 1: {str(e)}', "ERROR")
                        import traceback
                        traceback.print_exc()

                # Experiment 2: Prior Quality
                if RUN_EXPERIMENT_2:
                    try:
                        logger('', "INFO")
                        logger('EXPERIMENT 2: PRIOR MODEL QUALITY', "INFO")
                        logger('-' * 100, "INFO")

                        exp2_results = validators['exp2'].validate_split(split_id)
                        all_results['exp2'].append(exp2_results)

                        # Log to MLflow
                        mlflow.log_metrics({
                            'exp2_js_divergence': exp2_results['js_divergence_freq'],
                            'exp2_transition_frobenius_correlation': exp2_results['transition_frobenius_correlation'],
                            'exp2_transition_mean_abs_diff': exp2_results['transition_mean_abs_diff'],
                            'exp2_bigram_overlap': exp2_results['bigram_overlap_ratio']
                        })

                    except Exception as e:
                        logger(f'ERROR in Experiment 2: {str(e)}', "ERROR")
                        import traceback
                        traceback.print_exc()

                # Experiment 3: End-to-End
                if RUN_EXPERIMENT_3:
                    try:
                        logger('', "INFO")
                        logger('EXPERIMENT 3: END-TO-END SYNTHETIC DATA QUALITY', "INFO")
                        logger('-' * 100, "INFO")

                        exp3_results = validators['exp3'].validate_split(split_id)
                        all_results['exp3'].append(exp3_results)

                        # Log to MLflow
                        mlflow.log_metrics({
                            'exp3_mmd': exp3_results['mmd'],
                            'exp3_ks_rejection_rate': exp3_results['ks_rejection_rate'],
                            'exp3_corr_frobenius_correlation': exp3_results['corr_frobenius_correlation']
                        })

                    except Exception as e:
                        logger(f'ERROR in Experiment 3: {str(e)}', "ERROR")
                        import traceback
                        traceback.print_exc()

            split_duration = time.time() - split_start
            logger('', "INFO")
            logger(f'Split {split_id} complete in {split_duration:.1f}s', "INFO")

        # Compute aggregate statistics
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('AGGREGATE STATISTICS ACROSS SPLITS', "INFO")
        logger('=' * 100, "INFO")

        aggregate_stats = {}

        # Experiment 1 aggregates
        if len(all_results['exp1']) > 0:
            logger('', "INFO")
            logger('Experiment 1: VQ-VAE Reconstruction', "INFO")

            exp1_mse = compute_aggregate_statistics(all_results['exp1'], 'mse_overall')
            exp1_ks = compute_aggregate_statistics(all_results['exp1'], 'ks_rejection_rate')
            exp1_corr = compute_aggregate_statistics(all_results['exp1'], 'corr_frobenius_correlation')
            exp1_cosine = compute_aggregate_statistics(all_results['exp1'], 'cosine_similarity_mean')

            logger(f'  MSE: {exp1_mse["mean"]:.6f} ± {exp1_mse["std"]:.6f}', "INFO")
            logger(f'  KS Rejection Rate: {exp1_ks["mean"]:.4f} ± {exp1_ks["std"]:.4f}', "INFO")
            logger(f'  Corr Frobenius Correlation: {exp1_corr["mean"]:.6f} ± {exp1_corr["std"]:.6f}', "INFO")
            logger(f'  Cosine Similarity: {exp1_cosine["mean"]:.6f} ± {exp1_cosine["std"]:.6f}', "INFO")

            aggregate_stats['exp1'] = {
                'mse': exp1_mse,
                'ks_rejection_rate': exp1_ks,
                'corr_frobenius_correlation': exp1_corr,
                'cosine_similarity_mean': exp1_cosine
            }

            mlflow.log_metrics({
                'agg_exp1_mse_mean': exp1_mse['mean'],
                'agg_exp1_ks_rejection_mean': exp1_ks['mean'],
                'agg_exp1_corr_frobenius_correlation_mean': exp1_corr['mean'],
                'agg_exp1_cosine_similarity_mean': exp1_cosine['mean']
            })

        # Experiment 2 aggregates
        if len(all_results['exp2']) > 0:
            logger('', "INFO")
            logger('Experiment 2: Prior Quality', "INFO")

            exp2_js = compute_aggregate_statistics(all_results['exp2'], 'js_divergence_freq')
            exp2_trans = compute_aggregate_statistics(all_results['exp2'], 'transition_frobenius_correlation')
            exp2_trans_mad = compute_aggregate_statistics(all_results['exp2'], 'transition_mean_abs_diff')
            exp2_bigram = compute_aggregate_statistics(all_results['exp2'], 'bigram_overlap_ratio')

            logger(f'  JS Divergence: {exp2_js["mean"]:.6f} ± {exp2_js["std"]:.6f}', "INFO")
            logger(f'  Transition Frobenius Correlation: {exp2_trans["mean"]:.6f} ± {exp2_trans["std"]:.6f}', "INFO")
            logger(f'  Transition Mean Abs Diff: {exp2_trans_mad["mean"]:.6f} ± {exp2_trans_mad["std"]:.6f}', "INFO")
            logger(f'  Bigram Overlap: {exp2_bigram["mean"]:.4f} ± {exp2_bigram["std"]:.4f}', "INFO")

            aggregate_stats['exp2'] = {
                'js_divergence': exp2_js,
                'transition_frobenius_correlation': exp2_trans,
                'transition_mean_abs_diff': exp2_trans_mad,
                'bigram_overlap': exp2_bigram
            }

            mlflow.log_metrics({
                'agg_exp2_js_divergence_mean': exp2_js['mean'],
                'agg_exp2_transition_frobenius_correlation_mean': exp2_trans['mean'],
                'agg_exp2_transition_mean_abs_diff_mean': exp2_trans_mad['mean'],
                'agg_exp2_bigram_overlap_mean': exp2_bigram['mean']
            })

        # Experiment 3 aggregates
        if len(all_results['exp3']) > 0:
            logger('', "INFO")
            logger('Experiment 3: End-to-End Quality', "INFO")

            exp3_mmd = compute_aggregate_statistics(all_results['exp3'], 'mmd')
            exp3_ks = compute_aggregate_statistics(all_results['exp3'], 'ks_rejection_rate')
            exp3_corr = compute_aggregate_statistics(all_results['exp3'], 'corr_frobenius_correlation')

            logger(f'  MMD: {exp3_mmd["mean"]:.6f} ± {exp3_mmd["std"]:.6f}', "INFO")
            logger(f'  KS Rejection Rate: {exp3_ks["mean"]:.4f} ± {exp3_ks["std"]:.4f}', "INFO")
            logger(f'  Corr Frobenius Correlation: {exp3_corr["mean"]:.6f} ± {exp3_corr["std"]:.6f}', "INFO")

            aggregate_stats['exp3'] = {
                'mmd': exp3_mmd,
                'ks_rejection_rate': exp3_ks,
                'corr_frobenius_correlation': exp3_corr
            }

            mlflow.log_metrics({
                'agg_exp3_mmd_mean': exp3_mmd['mean'],
                'agg_exp3_ks_rejection_mean': exp3_ks['mean'],
                'agg_exp3_corr_frobenius_correlation_mean': exp3_corr['mean']
            })

        # Save all results to JSON
        results_file = OUTPUT_DIR / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'splits_validated': splits_to_run,
                'experiment_1_results': all_results['exp1'],
                'experiment_2_results': all_results['exp2'],
                'experiment_3_results': all_results['exp3'],
                'aggregate_statistics': aggregate_stats
            }, f, indent=2)

        logger('', "INFO")
        logger(f'Results saved to: {results_file}', "INFO")
        mlflow.log_artifact(str(results_file))

        # Summary
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('VALIDATION COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Splits validated: {len(splits_to_run)}', "INFO")
        logger(f'Results saved to: {OUTPUT_DIR}', "INFO")
        logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")


if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'

    start_time = time.time()

    try:
        main()

        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logger('', "INFO")
        logger(f'Total execution time: {hours}h {minutes}m {seconds}s', "INFO")
        logger('Stage 18 completed successfully', "INFO")

    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
