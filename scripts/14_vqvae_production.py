"""
VQ-VAE Production Training Script (Stage 13)

Trains final VQ-VAE models using the best hyperparameter configuration from Stage 12.
Generates latent representations for all samples in each split.

This is Stage 13 in the pipeline - follows hyperparameter search (Stage 12).

Input: Materialized split collections (split_0_input, split_1_input, ...)
       in database 'tfg' with standardized LOB vectors
       Best configuration from Stage 12 artifacts

Output: K trained models saved to artifacts/vqvae_models/production/
        K output collections (split_0_output, split_1_output, ...) with latent codes
        Collections renamed to split_X_input for next stage

Usage:
    python scripts/13_vqvae_production_training.py
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

import json
import torch
import mlflow
from datetime import datetime

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.vqvae_representation import (
    run_production_training,
    TRAINING_CONFIG
)

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"  # Read from feature standardization output (Stage 11)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "VQ-VAE_Production_Training"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
DRIVER_MEMORY = "8g"

# Artifact directories
ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts" / "vqvae_models"
HYPERPARAMETER_SEARCH_DIR = ARTIFACT_BASE_DIR / "hyperparameter_search"
PRODUCTION_DIR = ARTIFACT_BASE_DIR / "production"

# =================================================================================================
# Helper Functions
# =================================================================================================

def load_best_config() -> dict:
    """
    Load best hyperparameter configuration from Stage 12.
    
    Returns:
        Best configuration dictionary
    """
    config_path = HYPERPARAMETER_SEARCH_DIR / "best_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Best config not found at {config_path}. "
            f"Please run Stage 12 (hyperparameter search) first."
        )
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    logger(f'Loaded best config from: {config_path}', "INFO")
    logger(f'  Average validation loss: {config_data["avg_val_loss"]:.4f}', "INFO")
    logger(f'  Configuration: {config_data["config"]}', "INFO")
    
    return config_data["config"]

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 100, "INFO")
    logger('VQ-VAE PRODUCTION TRAINING (STAGE 13)', "INFO")
    logger('=' * 100, "INFO")
    
    # Load best configuration
    logger('', "INFO")
    logger('Loading best hyperparameter configuration from Stage 12...', "INFO")
    best_config = load_best_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger('', "INFO")
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
        app_name="VQVAEProductionTraining",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Display configuration
        logger('', "INFO")
        logger('Production training configuration:', "INFO")
        logger(f'  Database: {DB_NAME}', "INFO")
        logger(f'  Collection pattern: {COLLECTION_PREFIX}*{COLLECTION_SUFFIX}', "INFO")
        logger(f'  Production model directory: {PRODUCTION_DIR}', "INFO")
        
        logger('', "INFO")
        logger('Best hyperparameter configuration:', "INFO")
        for key, value in best_config.items():
            logger(f'  {key}: {value}', "INFO")
        
        logger('', "INFO")
        logger('Training configuration:', "INFO")
        for key, value in TRAINING_CONFIG.items():
            logger(f'  {key}: {value}', "INFO")
        
        # Run production training
        logger('', "INFO")
        logger('Starting production training...', "INFO")
        logger('', "INFO")
        
        results = run_production_training(
            spark=spark,
            db_name=DB_NAME,
            collection_prefix=COLLECTION_PREFIX,
            collection_suffix=COLLECTION_SUFFIX,
            device=device,
            best_config=best_config,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            production_dir=PRODUCTION_DIR
        )
        
        # Summary
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('PRODUCTION TRAINING COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Models trained: {results["num_splits"]}', "INFO")
        logger(f'Total samples processed: {results["total_samples"]:,}', "INFO")
        logger(f'Average validation loss: {results["avg_val_loss"]:.4f}', "INFO")
        logger(f'Models saved to: {PRODUCTION_DIR}', "INFO")
        logger(f'Latent collections: {COLLECTION_PREFIX}*_input (renamed)', "INFO")
        logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")
        
        logger('', "INFO")
        logger('Next steps:', "INFO")
        logger('  1. Review production models in MLflow UI', "INFO")
        logger('  2. Use latent representations in downstream models (Stage 14+)', "INFO")
        logger('  3. Each split has its own trained model and latent codes', "INFO")
        
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
        logger('Stage 13 completed successfully', "INFO")
        
    except Exception:
        sys.exit(1)