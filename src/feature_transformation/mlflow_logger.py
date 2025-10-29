"""
MLflow Logging for Feature Transformation Selection

Handles logging of transformation selection results to MLflow.
Saves artifacts to dedicated artifacts directory.
"""

import json
import os
import pandas as pd
import mlflow
from typing import Dict
from pathlib import Path

from src.utils.logging import logger

REPO_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "feature_transformation"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def log_split_results(split_id: int, results: Dict, train_sample_rate: float):
    """
    Log per-split transformation selection results to MLflow.
    
    Args:
        split_id: Split ID
        results: Results dictionary from processor
        train_sample_rate: Training data sampling rate used
    """
    with mlflow.start_run(run_name=f"split_{split_id}"):
        # Set tags
        mlflow.set_tag("split_id", split_id)
        mlflow.set_tag("stage", "validation")
        
        # Log parameters
        mlflow.log_param("split_id", split_id)
        mlflow.log_param("n_features", len(results))
        mlflow.log_param("train_sample_rate", train_sample_rate)
        
        # Log metrics per feature
        for feat_name, feat_results in results.items():
            if 'best_score' in feat_results:
                mlflow.log_metric(f"{feat_name}_best_pdf", feat_results['best_score'])
            
            # val_samples is now a dict per transform, log the max/average
            if 'val_samples' in feat_results and isinstance(feat_results['val_samples'], dict):
                if feat_results['val_samples']:
                    max_samples = max(feat_results['val_samples'].values())
                    mlflow.log_metric(f"{feat_name}_val_samples", max_samples)
        
        # Save artifacts
        _save_split_artifacts(split_id, results)
        
        logger(f'Logged split {split_id} results to MLflow', "INFO")


def _save_split_artifacts(split_id: int, results: Dict):
    """Save per-split artifacts to dedicated directory."""
    split_dir = ARTIFACTS_DIR / f"split_{split_id}"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Best transforms JSON
    best_transforms = {
        feat: res.get('best_transform', 'none')
        for feat, res in results.items()
    }
    
    filepath = split_dir / f'split_{split_id}_best_transforms.json'
    with open(filepath, 'w') as f:
        json.dump(best_transforms, f, indent=2)
    mlflow.log_artifact(str(filepath))
    
    # Validation scores CSV
    scores_data = []
    for feat_name, feat_results in results.items():
        for transform, score in feat_results.get('validation_scores', {}).items():
            scores_data.append({
                'feature': feat_name,
                'transform': transform,
                'p_df_score': score
            })
    
    if scores_data:
        scores_df = pd.DataFrame(scores_data)
        filepath = split_dir / f'split_{split_id}_validation_scores.csv'
        scores_df.to_csv(filepath, index=False)
        mlflow.log_artifact(str(filepath))
    
    # Fitted parameters JSON (optional, for reproducibility)
    fitted_params = {
        feat: res.get('fitted_params', {})
        for feat, res in results.items()
    }
    
    filepath = split_dir / f'split_{split_id}_fitted_params.json'
    with open(filepath, 'w') as f:
        json.dump(fitted_params, f, indent=2)
    mlflow.log_artifact(str(filepath))
    
    logger(f'Saved artifacts to {split_dir}', "INFO")


def log_aggregated_results(aggregated: Dict, final_transforms: Dict):
    """
    Log aggregated results across all splits to MLflow.
    
    Args:
        aggregated: Aggregated results dictionary
        final_transforms: Final selected transforms per feature
    """
    with mlflow.start_run(run_name="aggregation"):
        # Set tags
        mlflow.set_tag("stage", "final_selection")
        
        # Log metrics
        for feat_name, agg_res in aggregated.items():
            mlflow.log_metric(f"{feat_name}_stability", agg_res['stability'])
            mlflow.log_metric(f"{feat_name}_n_splits", agg_res['n_splits'])
        
        # Create aggregation directory
        agg_dir = ARTIFACTS_DIR / "aggregation"
        agg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final transforms
        filepath = agg_dir / 'final_transforms.json'
        with open(filepath, 'w') as f:
            json.dump(final_transforms, f, indent=2)
        mlflow.log_artifact(str(filepath))
        
        # Save frequency matrix
        freq_data = []
        for feat_name, agg_res in aggregated.items():
            for transform, count in agg_res['frequency_count'].items():
                freq_data.append({
                    'feature': feat_name,
                    'transform': transform,
                    'frequency': count,
                    'stability': agg_res['stability']
                })
        
        freq_df = pd.DataFrame(freq_data)
        filepath = agg_dir / 'transform_frequency.csv'
        freq_df.to_csv(filepath, index=False)
        mlflow.log_artifact(str(filepath))
        
        # Save average scores
        avg_scores_data = []
        for feat_name, agg_res in aggregated.items():
            for transform, avg_score in agg_res['avg_scores'].items():
                avg_scores_data.append({
                    'feature': feat_name,
                    'transform': transform,
                    'avg_p_df_score': avg_score
                })
        
        if avg_scores_data:
            avg_scores_df = pd.DataFrame(avg_scores_data)
            filepath = agg_dir / 'avg_validation_scores.csv'
            avg_scores_df.to_csv(filepath, index=False)
            mlflow.log_artifact(str(filepath))
        
        logger(f'Saved aggregated artifacts to {agg_dir}', "INFO")
        logger('Logged aggregated results to MLflow', "INFO")