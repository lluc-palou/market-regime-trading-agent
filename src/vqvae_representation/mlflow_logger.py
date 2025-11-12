"""
MLflow Logging Utilities

Handles logging of VQ-VAE training to MLflow:
- Hyperparameters
- Training/validation metrics
- Model artifacts
- Best configuration selection
"""

import mlflow
from typing import Dict, List
from pathlib import Path

from src.utils.logging import logger


def log_config_split_results(
    config_idx: int,
    split_id: int,
    config: Dict,
    results: Dict
):
    """
    Log results for a single config-split pair to MLflow.
    
    Args:
        config_idx: Configuration index
        split_id: Split ID
        config: Hyperparameter configuration
        results: Training results dictionary
    """
    # Log hyperparameters
    mlflow.log_param("config_id", config_idx)
    mlflow.log_param("split_id", split_id)
    for key, value in config.items():
        mlflow.log_param(key, value)
    
    # Log training metrics (at best epoch)
    train_losses = results['final_train_losses']
    for key, value in train_losses.items():
        if key not in ['num_batches', 'train_losses', 'val_losses', 'epoch']:
            mlflow.log_metric(f"train_{key}", value)
    
    # Log validation metrics (final)
    val_losses = results['final_val_losses']
    for key, value in val_losses.items():
        if key not in ['num_batches', 'num_samples']:
            mlflow.log_metric(f"val_{key}", value)
    
    # Log best metrics
    mlflow.log_metric("best_val_loss", results['best_val_loss'])
    mlflow.log_metric("best_epoch", results['best_epoch'])
    mlflow.log_metric("epochs_trained", results['epochs_trained'])


def log_config_aggregation(
    config_idx: int,
    config: Dict,
    split_results: List[Dict]
):
    """
    Log aggregated results across splits for a configuration.
    
    Args:
        config_idx: Configuration index
        config: Hyperparameter configuration
        split_results: List of results from each split
    """
    # Compute averages across splits
    avg_val_loss = sum(r['best_val_loss'] for r in split_results) / len(split_results)
    avg_val_perplexity = sum(r['final_val_losses']['perplexity'] for r in split_results) / len(split_results)
    avg_codebook_usage = sum(r['final_val_losses']['codebook_usage'] for r in split_results) / len(split_results)
    
    # Log aggregated metrics
    mlflow.log_metric("avg_val_loss", avg_val_loss)
    mlflow.log_metric("avg_val_perplexity", avg_val_perplexity)
    mlflow.log_metric("avg_codebook_usage", avg_codebook_usage)
    
    logger(
        f'Config {config_idx} aggregated: '
        f'avg_val_loss={avg_val_loss:.4f}, '
        f'avg_perplexity={avg_val_perplexity:.2f}, '
        f'avg_usage={avg_codebook_usage:.3f}',
        "INFO"
    )


def log_best_config(
    best_config: Dict,
    best_config_idx: int,
    avg_val_loss: float,
    avg_val_perplexity: float,
    avg_codebook_usage: float
):
    """
    Log best configuration to MLflow.
    
    Args:
        best_config: Best hyperparameter configuration
        best_config_idx: Index of best configuration
        avg_val_loss: Average validation loss
        avg_val_perplexity: Average perplexity
        avg_codebook_usage: Average codebook usage
    """
    # Log best configuration parameters
    for key, value in best_config.items():
        mlflow.log_param(f"best_{key}", value)
    
    # Log best metrics
    mlflow.log_metric("best_config_id", best_config_idx)
    mlflow.log_metric("best_avg_val_loss", avg_val_loss)
    mlflow.log_metric("best_avg_val_perplexity", avg_val_perplexity)
    mlflow.log_metric("best_avg_codebook_usage", avg_codebook_usage)
    
    logger('', "INFO")
    logger('=' * 80, "INFO")
    logger('BEST CONFIGURATION', "INFO")
    logger('=' * 80, "INFO")
    logger(f'Config ID: {best_config_idx}', "INFO")
    logger(f'Hyperparameters: {best_config}', "INFO")
    logger(f'Avg validation loss: {avg_val_loss:.4f}', "INFO")
    logger(f'Avg perplexity: {avg_val_perplexity:.2f}', "INFO")
    logger(f'Avg codebook usage: {avg_codebook_usage:.3f}', "INFO")


def log_search_summary(
    total_configs: int,
    total_splits: int,
    all_results: List[Dict]
):
    """
    Log summary of entire hyperparameter search.
    
    Args:
        total_configs: Total number of configurations tested
        total_splits: Total number of splits
        all_results: List of all configuration results
    """
    mlflow.log_param("total_configs_tested", total_configs)
    mlflow.log_param("total_splits", total_splits)
    mlflow.log_param("total_runs", total_configs * total_splits)
    
    # Log distribution of validation losses
    all_val_losses = [r['avg_val_loss'] for r in all_results]
    mlflow.log_metric("min_val_loss", min(all_val_losses))
    mlflow.log_metric("max_val_loss", max(all_val_losses))
    mlflow.log_metric("mean_val_loss", sum(all_val_losses) / len(all_val_losses))
    
    logger('', "INFO")
    logger('=' * 80, "INFO")
    logger('HYPERPARAMETER SEARCH SUMMARY', "INFO")
    logger('=' * 80, "INFO")
    logger(f'Total configurations tested: {total_configs}', "INFO")
    logger(f'Total splits: {total_splits}', "INFO")
    logger(f'Total runs: {total_configs * total_splits}', "INFO")
    logger(f'Validation loss range: [{min(all_val_losses):.4f}, {max(all_val_losses):.4f}]', "INFO")


def save_best_config_artifact(
    best_config: Dict,
    avg_val_loss: float,
    avg_val_perplexity: float,
    avg_codebook_usage: float,
    artifact_dir: Path
):
    """
    Save best configuration to YAML artifact.
    
    Args:
        best_config: Best hyperparameter configuration
        avg_val_loss: Average validation loss
        avg_val_perplexity: Average perplexity
        avg_codebook_usage: Average codebook usage
        artifact_dir: Directory to save artifact
    """
    import yaml
    from datetime import datetime
    
    # Create artifact directory
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare artifact data
    artifact_data = {
        'best_config': best_config,
        'metrics': {
            'avg_val_loss': float(avg_val_loss),
            'avg_val_perplexity': float(avg_val_perplexity),
            'avg_codebook_usage': float(avg_codebook_usage)
        },
        'search_date': datetime.now().isoformat(),
        'model_info': {
            'description': 'VQ-VAE for LOB representation learning',
            'input_dim': best_config['B'],
            'codebook_size': best_config['K'],
            'embedding_dim': best_config['D']
        }
    }
    
    # Save to YAML
    config_path = artifact_dir / 'best_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(artifact_data, f, default_flow_style=False)
    
    logger(f'Best configuration saved to: {config_path}', "INFO")
    
    # Log to MLflow
    mlflow.log_artifact(str(config_path))