"""
Production Trainer

Trains final VQ-VAE models for production deployment.

For each split:
1. Train model from scratch on ALL role='train' samples
2. Use early stopping with validation samples
3. Save best model to artifacts
4. Generate latent codes for ALL samples (train + validation)
5. Materialize to split_X_output collection
"""

import time
import torch
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from pymongo import MongoClient

from src.utils.logging import logger
from .model import VQVAEModel
from .trainer import VQVAETrainer
from .latent_generator import LatentGenerator
from .data_loader import discover_splits, get_split_info, get_all_hours


def run_production_training(
    spark,
    db_name: str,
    collection_prefix: str,
    collection_suffix: str,
    device,
    best_config: Dict,
    mlflow_experiment_name: str,
    production_dir: Path,
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    use_pymongo: bool = True,
    split_ids_filter: List[int] = None
) -> Dict:
    """
    Run complete production training pipeline.

    Process:
    1. Discover available splits
    2. For each split:
        - Train production model from scratch with early stopping
        - Save model artifacts
        - Generate latent codes for all samples
        - Write to split_X_output collection
    3. Rename collections: split_X_output -> split_X_input
    4. Log summary to MLflow

    Args:
        spark: SparkSession instance
        db_name: Database name
        collection_prefix: Prefix for split collections
        collection_suffix: Suffix for split collections
        device: torch device
        best_config: Best hyperparameter configuration from Stage 12
        mlflow_experiment_name: MLflow experiment name
        production_dir: Directory for production model artifacts
        mongo_uri: MongoDB connection URI (default: "mongodb://127.0.0.1:27017/")
        use_pymongo: Use PyMongo for 10-50× faster data loading (default: True)
        split_ids_filter: Optional list of specific split IDs to process (default: None = all splits)

    Returns:
        Dictionary with production training results
    """
    logger('=' * 100, "INFO")
    logger('VQ-VAE PRODUCTION TRAINING', "INFO")
    logger('=' * 100, "INFO")
    
    # Discover available splits
    split_ids = discover_splits(spark, db_name, collection_prefix, collection_suffix)

    if not split_ids:
        raise ValueError(
            f"No splits found in database '{db_name}' with pattern "
            f"'{collection_prefix}*{collection_suffix}'"
        )

    logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")

    # Apply split filter if provided
    if split_ids_filter is not None:
        original_count = len(split_ids)
        split_ids = [s for s in split_ids if s in split_ids_filter]
        logger(f'Filtered to {len(split_ids)} splits (from {original_count}): {split_ids}', "INFO")

        if not split_ids:
            raise ValueError(f"No splits remaining after applying filter: {split_ids_filter}")
    
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
    
    # MongoDB client for collection operations
    mongo_client = MongoClient("mongodb://127.0.0.1:27017/")
    db = mongo_client[db_name]
    
    # Track results across all splits
    all_split_results = []
    total_samples_processed = 0
    
    # Main MLflow run for entire production training
    with mlflow.start_run(run_name=f"vqvae_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log global parameters
        mlflow.log_param("db_name", db_name)
        mlflow.log_param("collection_pattern", f"{collection_prefix}*{collection_suffix}")
        mlflow.log_param("num_splits", len(split_ids))
        mlflow.log_param("device", str(device))
        
        # Log best config
        for key, value in best_config.items():
            mlflow.log_param(f"config_{key}", value)
        
        # Process each split
        for split_idx, split_id in enumerate(split_ids):
            split_collection = f"{collection_prefix}{split_id}{collection_suffix}"
            output_collection = f"{collection_prefix}{split_id}_output"
            
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
            
            # Train production model for this split
            with mlflow.start_run(
                run_name=f"split_{split_id}_production",
                nested=True
            ):
                logger('', "INFO")
                logger(f'Training production model for split {split_id}...', "INFO")
                
                # Train model
                model, training_results = train_production_model(
                    spark=spark,
                    db_name=db_name,
                    split_collection=split_collection,
                    device=device,
                    config=best_config,
                    all_hours=all_hours,
                    mongo_uri=mongo_uri,
                    use_pymongo=use_pymongo
                )
                
                # Log training results to MLflow
                mlflow.log_param("split_id", split_id)
                mlflow.log_param("split_collection", split_collection)
                mlflow.log_metric("best_val_loss", training_results['best_val_loss'])
                mlflow.log_metric("best_epoch", training_results['best_epoch'])
                mlflow.log_metric("epochs_trained", training_results['epochs_trained'])
                mlflow.log_metric("val_perplexity", training_results['final_val_losses']['perplexity'])
                mlflow.log_metric("val_codebook_usage", training_results['final_val_losses']['codebook_usage'])
                
                # Save model artifacts
                model_path = production_dir / f"split_{split_id}_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': best_config,
                    'split_id': split_id,
                    'training_results': training_results
                }, model_path)
                
                logger(f'Model saved to: {model_path}', "INFO")
                mlflow.log_artifact(str(model_path))
                
                # Generate latent representations
                logger('', "INFO")
                logger(f'Generating latent representations for split {split_id}...', "INFO")
                
                # Clear output collection if exists
                if output_collection in db.list_collection_names():
                    db[output_collection].drop()
                    logger(f'Cleared existing collection: {output_collection}', "INFO")
                
                # Generate and write latents
                latent_gen = LatentGenerator(
                    spark=spark,
                    db_name=db_name,
                    split_collection=split_collection,
                    output_collection=output_collection,
                    model=model,
                    device=device
                )
                
                latent_stats = latent_gen.generate_and_write_latents(
                    all_hours=all_hours,
                    split_id=split_id
                )
                
                latent_gen.close()
                
                # Log latent generation stats
                mlflow.log_metric("latent_total_samples", latent_stats['total_samples'])
                mlflow.log_metric("latent_train_samples", latent_stats['train_samples'])
                mlflow.log_metric("latent_val_samples", latent_stats['val_samples'])
                
                total_samples_processed += latent_stats['total_samples']
                
                # Store results
                split_duration = time.time() - split_start_time
                
                split_result = {
                    'split_id': split_id,
                    'training_results': training_results,
                    'latent_stats': latent_stats,
                    'model_path': str(model_path),
                    'duration_seconds': split_duration
                }
                
                all_split_results.append(split_result)
                
                logger('', "INFO")
                logger(f'Split {split_id} complete in {split_duration:.1f}s', "INFO")
                logger(f'  Best validation loss: {training_results["best_val_loss"]:.4f}', "INFO")
                logger(f'  Latent codes generated: {latent_stats["total_samples"]:,}', "INFO")
        
        # Rename collections: split_X_output -> split_X_input
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('RENAMING OUTPUT COLLECTIONS', "INFO")
        logger('=' * 100, "INFO")
        
        for split_id in split_ids:
            input_collection = f"{collection_prefix}{split_id}{collection_suffix}"
            output_collection = f"{collection_prefix}{split_id}_output"
            
            if output_collection in db.list_collection_names():
                # Drop old input collection
                if input_collection in db.list_collection_names():
                    db[input_collection].drop()
                    logger(f'Dropped old collection: {input_collection}', "INFO")
                
                # Rename output to input
                db[output_collection].rename(input_collection)
                logger(f'Renamed: {output_collection} -> {input_collection}', "INFO")
        
        # Save production summary
        summary_path = production_dir / 'production_summary.json'
        
        import json
        summary_data = {
            'training_date': datetime.now().isoformat(),
            'num_splits': len(split_ids),
            'total_samples_processed': total_samples_processed,
            'best_config': best_config,
            'split_results': [
                {
                    'split_id': r['split_id'],
                    'best_val_loss': float(r['training_results']['best_val_loss']),
                    'val_perplexity': float(r['training_results']['final_val_losses']['perplexity']),
                    'val_codebook_usage': float(r['training_results']['final_val_losses']['codebook_usage']),
                    'total_samples': r['latent_stats']['total_samples'],
                    'duration_seconds': r['duration_seconds']
                }
                for r in all_split_results
            ]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger(f'Production summary saved to: {summary_path}', "INFO")
        mlflow.log_artifact(str(summary_path))
        
        # Compute average validation loss across splits
        avg_val_loss = sum(r['training_results']['best_val_loss'] for r in all_split_results) / len(all_split_results)
        
        mlflow.log_metric("avg_val_loss_across_splits", avg_val_loss)
        mlflow.log_metric("total_samples_processed", total_samples_processed)
    
    # Close MongoDB client
    mongo_client.close()
    
    return {
        'num_splits': len(split_ids),
        'total_samples': total_samples_processed,
        'avg_val_loss': avg_val_loss,
        'split_results': all_split_results
    }


def train_production_model(
    spark,
    db_name: str,
    split_collection: str,
    device: torch.device,
    config: Dict,
    all_hours: List[datetime],
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    use_pymongo: bool = True
) -> tuple:
    """
    Train a single production model for one split.
    
    Uses the VQVAETrainer from Phase 1 but for production deployment.
    
    Args:
        spark: SparkSession instance
        db_name: Database name
        split_collection: Split collection name
        device: torch device
        config: Hyperparameter configuration
        all_hours: List of hourly time windows
        mongo_uri: MongoDB connection URI
        use_pymongo: Use PyMongo for fast data loading

    Returns:
        model: Trained VQVAEModel
        results: Training results dictionary
    """
    # Initialize trainer with PyMongo support for 10-50× faster loading
    trainer = VQVAETrainer(
        spark=spark,
        db_name=db_name,
        split_collection=split_collection,
        device=device,
        config=config,
        mongo_uri=mongo_uri,
        use_pymongo=use_pymongo
    )
    
    # Train with early stopping
    results = trainer.train_split(all_hours)
    
    # Return trained model and results
    return trainer.model, results