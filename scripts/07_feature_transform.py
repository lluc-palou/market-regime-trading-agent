"""
Feature Transformation Selection Script (Stage 07)

Selects optimal normalization transformations for LOB features using CPCV splits.

Input: split_X collections with 18 features
Processes: 16 features (excludes volatility and fwd_logret_1)
Output: Transformation selections for 16 features

Exclusions from transformation:
- volatility: Keep original scale (meaningful interpretation)
- fwd_logret_1: Target variable, keep original scale

Usage:
    python scripts/07_feature_transform.py
"""

import os
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# Unicode/MLflow Fix for Windows
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

import mlflow

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session
from src.feature_transformation import (
    FeatureTransformProcessor,
    aggregate_across_splits,
    select_final_transforms,
    identify_feature_names_from_collection  # FIXED: Use aggregation-based function
)
from src.feature_transformation.mlflow_logger import (
    log_split_results,
    log_aggregated_results
)

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
INPUT_COLLECTION_PREFIX = "split_"
INPUT_COLLECTION_SUFFIX = "_input"  # Split collections are named split_X_input

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Feature_Transformation"

MAX_SPLITS = 1
TRAIN_SAMPLE_RATE = 1.0

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
DRIVER_MEMORY = "4g"

# =================================================================================================
# Feature Filtering
# =================================================================================================

def filter_transformable_features(feature_names):
    """
    Filter features to only include those that should be transformed.
    
    Excludes:
    - volatility: Keep original scale (meaningful interpretation)
    - fwd_logret_*: Target variable, keep original scale
    
    Args:
        feature_names: List of all feature names (18 features from materialization)
        
    Returns:
        List of features to transform (16 features)
    """
    EXCLUDE_PATTERNS = [
        'fwd_logret_',      # Forward returns (targets)
    ]
    
    EXCLUDE_EXACT = []       # Keep original scale
    
    transformable = []
    
    for feat_name in feature_names:
        # Skip exact matches
        if feat_name in EXCLUDE_EXACT:
            continue
        
        # Skip pattern matches
        if any(feat_name.startswith(pattern) for pattern in EXCLUDE_PATTERNS):
            continue
        
        transformable.append(feat_name)
    
    excluded_count = len(feature_names) - len(transformable)
    logger(f'Filtered to {len(transformable)} transformable features (excluded {excluded_count})', "INFO")
    
    if excluded_count > 0:
        logger(f'Excluded from transformation: volatility, fwd_logret_1', "INFO")
    
    return transformable


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    log_section('FEATURE TRANSFORMATION SELECTION (STAGE 07)')
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark session
    logger('', "INFO")
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="FeatureTransformSelection",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Get feature names from first split using aggregation
        logger('', "INFO")
        logger('Identifying features...', "INFO")
        first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
        logger(f'Reading feature names from collection: {first_split}', "INFO")
        
        # FIXED: Use aggregation-based function to avoid Spark schema inference issues
        all_feature_names = identify_feature_names_from_collection(
            spark=spark,
            db_name=DB_NAME,
            collection=first_split
        )
        logger(f'Total features in split collections: {len(all_feature_names)}', "INFO")
        
        # Filter to transformable features only
        feature_names = filter_transformable_features(all_feature_names)
        logger(f'Processing {len(feature_names)} transformable features across {MAX_SPLITS} splits', "INFO")
        
        # Initialize processor
        processor = FeatureTransformProcessor(
            spark=spark,
            db_name=DB_NAME,
            input_collection_prefix=INPUT_COLLECTION_PREFIX,
            input_collection_suffix=INPUT_COLLECTION_SUFFIX,
            train_sample_rate=TRAIN_SAMPLE_RATE
        )
        
        # Process each split
        all_split_results = {}

        for split_id in range(MAX_SPLITS):
            # Process split - pass both filtered and full feature lists
            split_results = processor.process_split(
                split_id=split_id,
                feature_names=feature_names,  # Transformable features only
                all_feature_names=all_feature_names  # Full list for array validation
            )
            all_split_results[split_id] = split_results
            
            # Log to MLflow
            log_split_results(split_id, split_results, TRAIN_SAMPLE_RATE)
        
        # Aggregate results across splits
        logger('', "INFO")
        log_section('AGGREGATING RESULTS ACROSS SPLITS')
        aggregated = aggregate_across_splits(all_split_results)
        
        # Select final transformations
        logger('', "INFO")
        logger('Selecting final transformations...', "INFO")
        final_transforms = select_final_transforms(aggregated)
        
        # Log aggregated results
        log_aggregated_results(aggregated, final_transforms)
        
        # Save results
        results_dir = Path(REPO_ROOT) / 'artifacts' / 'feature_transformation'
        results_dir.mkdir(parents=True, exist_ok=True)

        import json
        results_file = results_dir / 'transformation_selection.json'
        with open(results_file, 'w') as f:
            json.dump({
                'final_transforms': final_transforms,
                'aggregated_metrics': {
                    feat: {
                        'selected_transform': final_transforms.get(feat, 'identity'),
                        'most_frequent_transform': agg['most_frequent_transform'],
                        'stability': float(agg['stability']),
                        'n_splits': agg['n_splits'],
                        'avg_scores': {k: float(v) for k, v in agg['avg_scores'].items()},
                        'frequency_count': agg['frequency_count']
                    }
                    for feat, agg in aggregated.items()
                }
            }, f, indent=2)
        
        logger(f'Results saved to: {results_file}', "INFO")
        
        log_section('TRANSFORMATION SELECTION COMPLETED')
        logger(f'Selected transformations for {len(final_transforms)} features', "INFO")
        logger(f'Excluded features (keep original scale): volatility, fwd_logret_1', "INFO")
        
    except Exception as e:
        logger(f'Error during transformation selection: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()


if __name__ == "__main__":
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    main()