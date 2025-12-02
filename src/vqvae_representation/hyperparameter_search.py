"""
Hyperparameter Search Orchestration

Coordinates grid search across:
- Multiple hyperparameter configurations
- Multiple CPCV splits

Logs all results to MLflow and selects best configuration.
"""

import mlflow
from itertools import product
from typing import Dict, List
from pathlib import Path
from datetime import datetime

from src.utils.logging import logger
from .trainer import VQVAETrainer
from .data_loader import discover_splits, get_split_info, get_all_hours
from .mlflow_logger import (
    log_config_split_results,
    log_config_aggregation,
    log_best_config,
    log_search_summary,
    save_best_config_artifact
)
from .config import HYPERPARAM_GRID, TRAINING_CONFIG


def generate_configs(grid: Dict) -> List[Dict]:
    """
    Generate all hyperparameter combinations from grid.
    
    Args:
        grid: Hyperparameter grid dictionary
        
    Returns:
        List of configuration dictionaries
    """
    keys = grid.keys()
    values = grid.values()
    combinations = list(product(*values))
    
    configs = [dict(zip(keys, combo)) for combo in combinations]
    
    return configs


def run_hyperparameter_search(
    spark,
    db_name: str,
    collection_prefix: str,
    collection_suffix: str,
    device,
    mlflow_experiment_name: str,
    artifact_base_dir: Path,
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    use_pymongo: bool = True
) -> Dict:
    """
    Run complete hyperparameter search.
    
    Process:
    1. Discover available splits
    2. Generate all hyperparameter configurations
    3. For each config:
        - For each split:
            - Train VQVAETrainer
            - Log results to MLflow
        - Aggregate results across splits
    4. Select best configuration
    5. Save best config to artifacts
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        collection_prefix: Prefix for split collections
        collection_suffix: Suffix for split collections
        device: torch device
        mlflow_experiment_name: MLflow experiment name
        artifact_base_dir: Base directory for artifacts
        
    Returns:
        Dictionary with:
            - best_config: Best hyperparameter configuration
            - best_config_idx: Index of best configuration
            - best_avg_val_loss: Best average validation loss
            - all_results: List of all configuration results
    """
    logger('=' * 100, "INFO")
    logger('VQ-VAE HYPERPARAMETER SEARCH', "INFO")
    logger('=' * 100, "INFO")
    
    # Discover available splits
    split_ids = discover_splits(spark, db_name, collection_prefix, collection_suffix)
    
    if not split_ids:
        raise ValueError(f"No splits found in database '{db_name}' with pattern '{collection_prefix}*{collection_suffix}'")
    
    logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")
    
    # Log info about each split
    for split_id in split_ids:
        split_collection = f"{collection_prefix}{split_id}{collection_suffix}"
        split_info = get_split_info(spark, db_name, split_collection)
        logger(
            f'  Split {split_id}: {split_info["total_samples"]:,} samples, '
            f'{split_info["min_timestamp"]} to {split_info["max_timestamp"]}',
            "INFO"
        )
    
    # Generate all configurations
    configs = generate_configs(HYPERPARAM_GRID)
    
    logger('', "INFO")
    logger(f'Hyperparameter grid:', "INFO")
    for key, values in HYPERPARAM_GRID.items():
        logger(f'  {key}: {values}', "INFO")
    logger(f'Total configurations: {len(configs)}', "INFO")
    logger(f'Total runs: {len(configs) * len(split_ids)}', "INFO")
    
    logger('', "INFO")
    logger(f'Training configuration:', "INFO")
    for key, value in TRAINING_CONFIG.items():
        logger(f'  {key}: {value}', "INFO")
    
    # Run grid search
    all_results = []
    
    with mlflow.start_run(run_name=f"vqvae_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log global parameters
        mlflow.log_param("db_name", db_name)
        mlflow.log_param("collection_pattern", f"{collection_prefix}*{collection_suffix}")
        mlflow.log_param("num_splits", len(split_ids))
        mlflow.log_param("num_configs", len(configs))
        mlflow.log_param("device", str(device))
        
        for key, value in TRAINING_CONFIG.items():
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
                
                # Get hours for this split
                all_hours = get_all_hours(spark, db_name, split_collection)
                
                if not all_hours:
                    logger(f'No hours found in {split_collection}, skipping', "WARNING")
                    continue
                
                # Train model
                with mlflow.start_run(
                    run_name=f"config{config_idx}_split{split_id}",
                    nested=True
                ):
                    trainer = VQVAETrainer(
                        spark=spark,
                        db_name=db_name,
                        split_collection=split_collection,
                        device=device,
                        config=config,
                        mongo_uri=mongo_uri,
                        use_pymongo=use_pymongo
                    )
                    
                    result = trainer.train_split(all_hours)
                    
                    # Log to MLflow
                    log_config_split_results(config_idx, split_id, config, result)
                    
                    config_results.append(result)
            
            # Aggregate results across splits
            if config_results:
                avg_val_loss = sum(r['best_val_loss'] for r in config_results) / len(config_results)
                avg_val_perplexity = sum(r['final_val_losses']['perplexity'] for r in config_results) / len(config_results)
                avg_codebook_usage = sum(r['final_val_losses']['codebook_usage'] for r in config_results) / len(config_results)
                
                all_results.append({
                    'config_idx': config_idx,
                    'config': config,
                    'avg_val_loss': avg_val_loss,
                    'avg_val_perplexity': avg_val_perplexity,
                    'avg_codebook_usage': avg_codebook_usage,
                    'split_results': config_results
                })
                
                # Log aggregated results
                with mlflow.start_run(
                    run_name=f"config{config_idx}_aggregated",
                    nested=True
                ):
                    for key, value in config.items():
                        mlflow.log_param(key, value)
                    log_config_aggregation(config_idx, config, config_results)
        
        # Select best configuration
        if not all_results:
            raise ValueError("No valid results from hyperparameter search!")
        
        best_result = min(all_results, key=lambda x: x['avg_val_loss'])
        
        # Log best configuration
        log_best_config(
            best_config=best_result['config'],
            best_config_idx=best_result['config_idx'],
            avg_val_loss=best_result['avg_val_loss'],
            avg_val_perplexity=best_result['avg_val_perplexity'],
            avg_codebook_usage=best_result['avg_codebook_usage']
        )
        
        # Log search summary
        log_search_summary(len(configs), len(split_ids), all_results)
        
        # Save best config to artifacts
        artifact_dir = artifact_base_dir / 'hyperparameter_search'
        save_best_config_artifact(
            best_config=best_result['config'],
            avg_val_loss=best_result['avg_val_loss'],
            avg_val_perplexity=best_result['avg_val_perplexity'],
            avg_codebook_usage=best_result['avg_codebook_usage'],
            artifact_dir=artifact_dir
        )
        
        # Also save all results as JSON
        import json
        results_path = artifact_dir / 'all_results.json'
        with open(results_path, 'w') as f:
            # Convert to JSON-serializable format
            results_json = {
                'search_date': datetime.now().isoformat(),
                'total_configs': len(configs),
                'total_splits': len(split_ids),
                'best_config_idx': best_result['config_idx'],
                'best_avg_val_loss': float(best_result['avg_val_loss']),
                'all_results': [
                    {
                        'config_idx': r['config_idx'],
                        'config': r['config'],
                        'avg_val_loss': float(r['avg_val_loss']),
                        'avg_val_perplexity': float(r['avg_val_perplexity']),
                        'avg_codebook_usage': float(r['avg_codebook_usage'])
                    }
                    for r in all_results
                ]
            }
            json.dump(results_json, f, indent=2)
        
        mlflow.log_artifact(str(results_path))
        
        logger(f'All results saved to: {results_path}', "INFO")
    
    return {
        'best_config': best_result['config'],
        'best_config_idx': best_result['config_idx'],
        'best_avg_val_loss': best_result['avg_val_loss'],
        'all_results': all_results
    }