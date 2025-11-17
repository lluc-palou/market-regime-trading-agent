"""
VQ-VAE Hyperparameter Search Script (Stage 12)

Performs grid search across hyperparameter configurations and CPCV splits
to select the optimal VQ-VAE architecture for LOB representation learning.

This is Stage 12 in the pipeline - follows feature standardization (Stage 11).

Input: Materialized split collections (split_0_input, split_1_input, ...)
        in database 'tfg' with standardized LOB vectors

Output: Best hyperparameter configuration saved to artifacts/vqvae_models/

Usage:
    python scripts/12_vqvae_hyperparam_search.py
"""

import os
import sys
from pathlib import Path

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

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.vqvae_representation import (
    run_hyperparameter_search,
    HYPERPARAM_GRID,
    TRAINING_CONFIG
)

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"  # Read from feature standardization output (Stage 11)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "VQ-VAE_Hyperparameter_Search"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "8g"

# Artifact directory
ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts" / "vqvae_models"

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 100, "INFO")
    logger('VQ-VAE HYPERPARAMETER SEARCH (STAGE 12)', "INFO")
    logger('=' * 100, "INFO")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger(f'Device: {device}', "INFO")
    
    if device.type == 'cuda':
        logger(f'CUDA Device: {torch.cuda.get_device_name(0)}', "INFO")
        logger(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB', "INFO")
    
    # Setup MLflow
    logger('', "INFO")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}', "INFO")
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark session
    logger('', "INFO")
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="VQVAEHyperparameterSearch",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Display search configuration
        logger('', "INFO")
        logger('Search configuration:', "INFO")
        logger(f'  Database: {DB_NAME}', "INFO")
        logger(f'  Collection pattern: {COLLECTION_PREFIX}*{COLLECTION_SUFFIX}', "INFO")
        logger(f'  Artifact directory: {ARTIFACT_BASE_DIR}', "INFO")
        
        logger('', "INFO")
        logger('Hyperparameter grid:', "INFO")
        for key, values in HYPERPARAM_GRID.items():
            logger(f'  {key}: {values}', "INFO")
        
        logger('', "INFO")
        logger('Training configuration:', "INFO")
        for key, value in TRAINING_CONFIG.items():
            logger(f'  {key}: {value}', "INFO")
        
        # Run hyperparameter search
        logger('', "INFO")
        logger('Starting hyperparameter search...', "INFO")
        logger('', "INFO")
        
        results = run_hyperparameter_search(
            spark=spark,
            db_name=DB_NAME,
            collection_prefix=COLLECTION_PREFIX,
            collection_suffix=COLLECTION_SUFFIX,
            device=device,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            artifact_base_dir=ARTIFACT_BASE_DIR
        )
        
        # Summary
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('HYPERPARAMETER SEARCH COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Best configuration (ID {results["best_config_idx"]}): {results["best_config"]}', "INFO")
        logger(f'Best average validation loss: {results["best_avg_val_loss"]:.4f}', "INFO")
        logger(f'Results saved to: {ARTIFACT_BASE_DIR / "hyperparameter_search"}', "INFO")
        logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")
        
        logger('', "INFO")
        logger('Next steps:', "INFO")
        logger('  1. Review results in MLflow UI', "INFO")
        logger('  2. Run Stage 13 (production training) with selected configuration', "INFO")
        logger('  3. Generate latent representations for downstream models', "INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        
        if mlflow.active_run():
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_message", str(e))
            mlflow.end_run(status="FAILED")
        
        raise
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    
    import time
    start_time = time.time()
    
    try:
        main()
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logger('', "INFO")
        logger(f'Total execution time: {hours}h {minutes}m {seconds}s', "INFO")
        logger('Stage 12 completed successfully', "INFO")
        
    except Exception:
        sys.exit(1)