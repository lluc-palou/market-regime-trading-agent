"""
MLflow Logging for EWMA Half-Life Selection

Handles logging of half-life selection results to MLflow.
Saves artifacts to dedicated artifacts directory.
"""

import json
import os
import pandas as pd
import mlflow
from typing import Dict
from pathlib import Path

from src.utils.logging import logger

# Artifacts directory at repository root level (same level as scripts/, src/)
# Path: src/feature_standardization/mlflow_logger.py -> src/ -> rdl-lob/
# artifacts/ is at rdl-lob/artifacts/
REPO_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ewma_halflife_selection"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def log_split_results(split_id: int, results: Dict, train_sample_rate: float):
    """
    Log per-split half-life selection results to MLflow.
    
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
            
            if 'best_half_life' in feat_results:
                mlflow.log_metric(f"{feat_name}_best_halflife", feat_results['best_half_life'])
            
            # Log validation sample counts
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
    
    # Best half-lives JSON
    best_half_lives = {
        feat: res.get('best_half_life', 20)
        for feat, res in results.items()
    }
    
    filepath = split_dir / f'split_{split_id}_best_halflifes.json'
    with open(filepath, 'w') as f:
        json.dump(best_half_lives, f, indent=2)
    mlflow.log_artifact(str(filepath))
    
    # Validation scores CSV
    scores_data = []
    for feat_name, feat_results in results.items():
        for half_life, score in feat_results.get('validation_scores', {}).items():
            scores_data.append({
                'feature': feat_name,
                'half_life': half_life,
                'p_df_score': score
            })
    
    if scores_data:
        scores_df = pd.DataFrame(scores_data)
        filepath = split_dir / f'split_{split_id}_validation_scores.csv'
        scores_df.to_csv(filepath, index=False)
        mlflow.log_artifact(str(filepath))
    
    # Fitted parameters JSON (EWMA statistics)
    fitted_params = {}
    for feat, res in results.items():
        if 'fitted_params' in res:
            # Convert to serializable format
            fitted_params[feat] = {
                int(hl): params 
                for hl, params in res['fitted_params'].items()
            }
    
    filepath = split_dir / f'split_{split_id}_fitted_params.json'
    with open(filepath, 'w') as f:
        json.dump(fitted_params, f, indent=2)
    mlflow.log_artifact(str(filepath))
    
    logger(f'Saved artifacts to {split_dir}', "INFO")


def log_aggregated_results(aggregated: Dict, final_half_lives: Dict):
    """
    Log aggregated results across all splits to MLflow.
    
    Args:
        aggregated: Aggregated results dictionary
        final_half_lives: Final selected half-lives per feature
    """
    with mlflow.start_run(run_name="aggregation"):
        # Set tags
        mlflow.set_tag("stage", "final_selection")
        
        # Log metrics
        for feat_name, agg_res in aggregated.items():
            mlflow.log_metric(f"{feat_name}_stability", agg_res['stability'])
            mlflow.log_metric(f"{feat_name}_n_splits", agg_res['n_splits'])
            mlflow.log_metric(f"{feat_name}_final_halflife", final_half_lives[feat_name])
        
        # Create aggregation directory
        agg_dir = ARTIFACTS_DIR / "aggregation"
        agg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final half-lives
        filepath = agg_dir / 'final_halflifes.json'
        with open(filepath, 'w') as f:
            json.dump(final_half_lives, f, indent=2)
        mlflow.log_artifact(str(filepath))
        
        # Save frequency matrix
        freq_data = []
        for feat_name, agg_res in aggregated.items():
            for half_life, count in agg_res['frequency_count'].items():
                freq_data.append({
                    'feature': feat_name,
                    'half_life': half_life,
                    'frequency': count,
                    'stability': agg_res['stability']
                })
        
        freq_df = pd.DataFrame(freq_data)
        filepath = agg_dir / 'halflife_frequency.csv'
        freq_df.to_csv(filepath, index=False)
        mlflow.log_artifact(str(filepath))
        
        # Save average scores
        avg_scores_data = []
        for feat_name, agg_res in aggregated.items():
            for half_life, avg_score in agg_res['avg_scores'].items():
                avg_scores_data.append({
                    'feature': feat_name,
                    'half_life': half_life,
                    'avg_p_df_score': avg_score
                })
        
        if avg_scores_data:
            avg_scores_df = pd.DataFrame(avg_scores_data)
            filepath = agg_dir / 'avg_validation_scores.csv'
            avg_scores_df.to_csv(filepath, index=False)
            mlflow.log_artifact(str(filepath))
        
        logger(f'Saved aggregated artifacts to {agg_dir}', "INFO")
        logger('Logged aggregated results to MLflow', "INFO")