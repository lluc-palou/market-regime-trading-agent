"""
Prior Hyperparameter Search Orchestration

Coordinates grid search for prior model architecture.
"""

import json
import mlflow
from itertools import product
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.utils.logging import logger
from src.vqvae_representation.data_loader import discover_splits, get_all_hours
from .prior_trainer import PriorTrainer
from .prior_config import PRIOR_HYPERPARAM_GRID, PRIOR_TRAINING_CONFIG


def generate_configs(grid: Dict) -> List[Dict]:
    """Generate all hyperparameter combinations from grid."""
    keys = grid.keys()
    values = grid.values()
    combinations = list(product(*values))
    
    configs = [dict(zip(keys, combo)) for combo in combinations]
    
    return configs


def run_prior_hyperparameter_search(
    spark,
    db_name: str,
    collection_prefix: str,
    collection_suffix: str,
    device,
    codebook_size: int,
    mlflow_experiment_name: str,
    artifact_base_dir: Path
) -> Dict:
    """
    Run prior hyperparameter search.
    
    Args:
        spark: SparkSession
        db_name: Database name
        collection_prefix: Split collection prefix
        collection_suffix: Split collection suffix
        device: torch device
        codebook_size: VQ-VAE codebook size (K)
        mlflow_experiment_name: MLflow experiment name
        artifact_base_dir: Base directory for artifacts
        
    Returns:
        Dictionary with best configuration and results
    """
    logger('=' * 100, "INFO")
    logger('PRIOR HYPERPARAMETER SEARCH', "INFO")
    logger('=' * 100, "INFO")
    
    # Discover splits
    split_ids = discover_splits(spark, db_name, collection_prefix, collection_suffix)
    
    if not split_ids:
        raise ValueError(f"No splits found with pattern '{collection_prefix}*{collection_suffix}'")
    
    logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")
    
    # Generate configurations
    configs = generate_configs(PRIOR_HYPERPARAM_GRID)
    
    logger('', "INFO")
    logger(f'Hyperparameter grid:', "INFO")
    for key, values in PRIOR_HYPERPARAM_GRID.items():
        logger(f'  {key}: {values}', "INFO")
    logger(f'Total configurations: {len(configs)}', "INFO")
    logger(f'Total runs: {len(configs) * len(split_ids)}', "INFO")
    
    logger('', "INFO")
    logger(f'Training configuration:', "INFO")
    for key, value in PRIOR_TRAINING_CONFIG.items():
        logger(f'  {key}: {value}', "INFO")
    
    # Run grid search
    all_results = []
    
    with mlflow.start_run(run_name=f"prior_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log global parameters
        mlflow.log_param("db_name", db_name)
        mlflow.log_param("num_splits", len(split_ids))
        mlflow.log_param("num_configs", len(configs))
        mlflow.log_param("codebook_size", codebook_size)
        
        for key, value in PRIOR_TRAINING_CONFIG.items():
            mlflow.log_param(f"training_{key}", value)
        
        # Process each configuration
        for config_idx, config in enumerate(configs):
            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger(f'CONFIGURATION {config_idx+1}/{len(configs)}', "INFO")
            logger('=' * 100, "INFO")
            logger(f'Hyperparameters: {config}', "INFO")
            
            config_results = []
            
            # Process each split
            for split_id in split_ids:
                split_collection = f"{collection_prefix}{split_id}{collection_suffix}"
                
                logger('', "INFO")
                logger(f'Training on {split_collection}...', "INFO")
                
                # Get hours
                all_hours = get_all_hours(spark, db_name, split_collection)
                
                if not all_hours:
                    logger(f'No hours found in {split_collection}, skipping', "WARNING")
                    continue
                
                # Train model
                with mlflow.start_run(
                    run_name=f"config{config_idx}_split{split_id}",
                    nested=True
                ):
                    trainer = PriorTrainer(
                        spark=spark,
                        db_name=db_name,
                        split_collection=split_collection,
                        device=device,
                        config=config,
                        codebook_size=codebook_size
                    )
                    
                    result = trainer.train_split(all_hours)
                    
                    if result is None:
                        logger(f'Training failed for {split_collection}', "WARNING")
                        continue
                    
                    # Log to MLflow
                    mlflow.log_param("split_id", split_id)
                    mlflow.log_param("config_idx", config_idx)
                    for key, value in config.items():
                        mlflow.log_param(key, value)

                    mlflow.log_metric("best_val_loss", result['best_val_loss'])
                    mlflow.log_metric("best_epoch", result['best_epoch'])
                    mlflow.log_metric("epochs_trained", result['epochs_trained'])

                    # Log loss components for analysis
                    if result.get('final_val_codebook_loss') is not None:
                        mlflow.log_metric("final_val_codebook_loss", result['final_val_codebook_loss'])
                        mlflow.log_metric("final_val_target_loss", result['final_val_target_loss'])
                        mlflow.log_metric("final_train_codebook_loss", result['final_train_codebook_loss'])
                        mlflow.log_metric("final_train_target_loss", result['final_train_target_loss'])
                    
                    config_results.append(result)
            
            # Aggregate results across splits
            if config_results:
                avg_val_loss = sum(r['best_val_loss'] for r in config_results) / len(config_results)
                
                all_results.append({
                    'config_idx': config_idx,
                    'config': config,
                    'avg_val_loss': avg_val_loss,
                    'split_results': config_results
                })
                
                # Log aggregated
                with mlflow.start_run(
                    run_name=f"config{config_idx}_aggregated",
                    nested=True
                ):
                    mlflow.log_param("config_idx", config_idx)
                    for key, value in config.items():
                        mlflow.log_param(key, value)
                    mlflow.log_metric("avg_val_loss", avg_val_loss)
                    mlflow.log_metric("num_splits", len(config_results))
        
        # Select best configuration
        if not all_results:
            raise ValueError("No valid results from hyperparameter search!")
        
        best_result = min(all_results, key=lambda x: x['avg_val_loss'])
        
        # Log best config
        mlflow.log_metric("best_avg_val_loss", best_result['avg_val_loss'])
        mlflow.log_param("best_config_idx", best_result['config_idx'])
        for key, value in best_result['config'].items():
            mlflow.log_param(f"best_{key}", value)
        
        # Save best config
        artifact_dir = artifact_base_dir / 'hyperparameter_search'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        best_config_path = artifact_dir / 'best_prior_config.json'
        with open(best_config_path, 'w') as f:
            json.dump({
                'config': best_result['config'],
                'avg_val_loss': float(best_result['avg_val_loss']),
                'search_date': datetime.now().isoformat()
            }, f, indent=2)
        
        mlflow.log_artifact(str(best_config_path))
        
        # Save all results
        results_path = artifact_dir / 'all_prior_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'search_date': datetime.now().isoformat(),
                'total_configs': len(configs),
                'total_splits': len(split_ids),
                'codebook_size': codebook_size,
                'best_config_idx': best_result['config_idx'],
                'best_avg_val_loss': float(best_result['avg_val_loss']),
                'all_results': [
                    {
                        'config_idx': r['config_idx'],
                        'config': r['config'],
                        'avg_val_loss': float(r['avg_val_loss'])
                    }
                    for r in all_results
                ]
            }, f, indent=2)
        
        mlflow.log_artifact(str(results_path))
        
        logger(f'Best config saved to: {best_config_path}', "INFO")
    
    return {
        'best_config': best_result['config'],
        'best_config_idx': best_result['config_idx'],
        'best_avg_val_loss': best_result['avg_val_loss'],
        'all_results': all_results
    }