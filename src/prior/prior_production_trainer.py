"""
Prior Production Trainer

Trains final Prior models for production deployment.

For each split:
1. Train Prior model from scratch on ALL role='train' samples
2. Use early stopping with validation samples
3. Save best model to artifacts/prior_models/production/split_X/
4. Log metrics to MLflow

This module follows the same pattern as VQ-VAE production training (Stage 14).
"""

import time
import torch
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.utils.logging import logger
from .prior_model import LatentPriorCNN
from .prior_trainer import PriorTrainer
from src.vqvae_representation.data_loader import discover_splits, get_split_info, get_all_hours


def train_production_model(
    spark,
    db_name: str,
    split_collection: str,
    device,
    config: Dict,
    codebook_size: int,
    all_hours: List[datetime]
) -> tuple:
    """
    Train a production Prior model for a single split.

    Args:
        spark: SparkSession
        db_name: Database name
        split_collection: Split collection name
        device: torch device
        config: Hyperparameter configuration
        codebook_size: VQ-VAE codebook size
        all_hours: List of hourly time windows

    Returns:
        Tuple of (trained_model, training_results)
    """
    # Initialize trainer
    trainer = PriorTrainer(
        spark=spark,
        db_name=db_name,
        split_collection=split_collection,
        device=device,
        config=config,
        codebook_size=codebook_size
    )

    # Train
    training_results = trainer.train_split(all_hours)

    if training_results is None:
        raise ValueError(f"Training failed for {split_collection}")

    return trainer.model, training_results


def run_production_training(
    spark,
    db_name: str,
    collection_prefix: str,
    collection_suffix: str,
    device,
    best_config: Dict,
    codebook_size: int,
    mlflow_experiment_name: str,
    production_dir: Path
) -> Dict:
    """
    Run complete Prior production training pipeline.

    Process:
    1. Discover available splits
    2. For each split:
        - Train production Prior model from scratch with early stopping
        - Save model artifacts to production directory
        - Log metrics to MLflow
    3. Return aggregate statistics

    Args:
        spark: SparkSession instance
        db_name: Database name
        collection_prefix: Prefix for split collections
        collection_suffix: Suffix for split collections
        device: torch device
        best_config: Best hyperparameter configuration from Stage 15
        codebook_size: VQ-VAE codebook size (K)
        mlflow_experiment_name: MLflow experiment name
        production_dir: Directory for production model artifacts

    Returns:
        Dictionary with production training results
    """
    logger('=' * 100, "INFO")
    logger('PRIOR PRODUCTION TRAINING', "INFO")
    logger('=' * 100, "INFO")

    # Discover available splits
    split_ids = discover_splits(spark, db_name, collection_prefix, collection_suffix)

    if not split_ids:
        raise ValueError(
            f"No splits found in database '{db_name}' with pattern "
            f"'{collection_prefix}*{collection_suffix}'"
        )

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

    # Create production directory
    production_dir.mkdir(parents=True, exist_ok=True)

    # Track results across all splits
    all_split_results = []

    # Main MLflow run for entire production training
    with mlflow.start_run(run_name=f"prior_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log global parameters
        mlflow.log_param("db_name", db_name)
        mlflow.log_param("collection_pattern", f"{collection_prefix}*{collection_suffix}")
        mlflow.log_param("num_splits", len(split_ids))
        mlflow.log_param("device", str(device))
        mlflow.log_param("codebook_size", codebook_size)

        # Log best config
        for key, value in best_config.items():
            mlflow.log_param(f"config_{key}", value)

        # Process each split
        for split_idx, split_id in enumerate(split_ids):
            split_collection = f"{collection_prefix}{split_id}{collection_suffix}"

            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger(f'SPLIT {split_id} ({split_idx+1}/{len(split_ids)})', "INFO")
            logger('=' * 100, "INFO")

            split_start_time = time.time()

            # Get hours for this split
            all_hours = get_all_hours(spark, db_name, split_collection)

            if not all_hours:
                logger(f'No hours found in {split_collection}, skipping', "WARNING")
                continue

            logger(f'Found {len(all_hours)} hourly windows', "INFO")

            # Train production model for this split
            with mlflow.start_run(
                run_name=f"split_{split_id}_production",
                nested=True
            ):
                logger('', "INFO")
                logger(f'Training production Prior model for split {split_id}...', "INFO")

                # Train model
                model, training_results = train_production_model(
                    spark=spark,
                    db_name=db_name,
                    split_collection=split_collection,
                    device=device,
                    config=best_config,
                    codebook_size=codebook_size,
                    all_hours=all_hours
                )

                # Log training results to MLflow
                mlflow.log_param("split_id", split_id)
                mlflow.log_param("split_collection", split_collection)
                mlflow.log_metric("best_val_loss", training_results['best_val_loss'])
                mlflow.log_metric("best_epoch", training_results['best_epoch'])
                mlflow.log_metric("epochs_trained", training_results['epochs_trained'])
                if training_results['final_train_loss'] is not None:
                    mlflow.log_metric("final_train_loss", training_results['final_train_loss'])

                # Create split-specific directory
                split_dir = production_dir / f"split_{split_id}"
                split_dir.mkdir(parents=True, exist_ok=True)

                # Save model artifacts
                model_path = split_dir / "prior_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': best_config,
                    'codebook_size': codebook_size,
                    'split_id': split_id,
                    'training_results': training_results,
                    'timestamp': datetime.now().isoformat()
                }, model_path)

                logger(f'Model saved to: {model_path}', "INFO")
                mlflow.log_artifact(str(model_path))

                # Track results
                split_result = {
                    'split_id': split_id,
                    'best_val_loss': training_results['best_val_loss'],
                    'best_epoch': training_results['best_epoch'],
                    'epochs_trained': training_results['epochs_trained'],
                    'model_path': str(model_path)
                }
                all_split_results.append(split_result)

                split_duration = time.time() - split_start_time
                logger('', "INFO")
                logger(f'Split {split_id} complete in {split_duration:.1f}s', "INFO")
                logger(f'  Best validation loss: {training_results["best_val_loss"]:.4f}', "INFO")
                logger(f'  Best epoch: {training_results["best_epoch"]}', "INFO")
                logger(f'  Model saved to: {model_path}', "INFO")

        # Compute aggregate statistics
        if all_split_results:
            avg_val_loss = sum(r['best_val_loss'] for r in all_split_results) / len(all_split_results)
            avg_epochs = sum(r['epochs_trained'] for r in all_split_results) / len(all_split_results)

            mlflow.log_metric("avg_val_loss", avg_val_loss)
            mlflow.log_metric("avg_epochs_trained", avg_epochs)
            mlflow.log_metric("total_splits_trained", len(all_split_results))

            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger('PRODUCTION TRAINING SUMMARY', "INFO")
            logger('=' * 100, "INFO")
            logger(f'Splits trained: {len(all_split_results)}', "INFO")
            logger(f'Average validation loss: {avg_val_loss:.4f}', "INFO")
            logger(f'Average epochs trained: {avg_epochs:.1f}', "INFO")
            logger(f'Models saved to: {production_dir}', "INFO")
        else:
            logger('No splits were successfully trained!', "ERROR")
            raise ValueError("Production training failed for all splits")

    return {
        'num_splits': len(all_split_results),
        'avg_val_loss': avg_val_loss,
        'avg_epochs_trained': avg_epochs,
        'split_results': all_split_results,
        'production_dir': str(production_dir)
    }
