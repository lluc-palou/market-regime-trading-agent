"""
Prior Production Training Script (Stage 14b)

Trains final prior models for each split using best configuration.

Input: best_prior_config.json + split_X_input collections
Output: split_X_prior.pth models

Usage:
    python scripts/14b_prior_production_training.py
"""

import os
import sys
import json
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Unicode/MLflow fix
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

import torch
import mlflow

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.prior.prior_production_training import run_prior_production_training

# Configuration
DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LOB_Prior_Production_Training"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
DRIVER_MEMORY = "8g"

ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts" / "prior_models"
PRIOR_SEARCH_DIR = ARTIFACT_BASE_DIR / "hyperparameter_search"
PRODUCTION_DIR = ARTIFACT_BASE_DIR / "production"

VQVAE_CONFIG_PATH = Path(REPO_ROOT) / "artifacts" / "vqvae_models" / "hyperparameter_search" / "best_config.json"


def load_best_prior_config():
    """Load best prior configuration from Stage 14a."""
    config_path = PRIOR_SEARCH_DIR / "best_prior_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Best prior config not found at {config_path}. "
            f"Please run Stage 14a first."
        )
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    logger(f'Loaded best prior config from: {config_path}', "INFO")
    logger(f'  Avg validation loss: {config_data["avg_val_loss"]:.4f}', "INFO")
    logger(f'  Configuration: {config_data["config"]}', "INFO")
    
    return config_data["config"]


def main():
    logger('=' * 100, "INFO")
    logger('PRIOR PRODUCTION TRAINING (STAGE 14b)', "INFO")
    logger('=' * 100, "INFO")
    
    # Load best prior config
    best_config = load_best_prior_config()
    
    # Load VQ-VAE config for codebook size
    if not VQVAE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"VQ-VAE config not found: {VQVAE_CONFIG_PATH}")
    
    with open(VQVAE_CONFIG_PATH, 'r') as f:
        vqvae_config = json.load(f)
        codebook_size = vqvae_config['config']['K']
    
    logger(f'VQ-VAE codebook size: {codebook_size}', "INFO")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger(f'Device: {device}', "INFO")
    
    if device.type == 'cuda':
        logger(f'CUDA Device: {torch.cuda.get_device_name(0)}', "INFO")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="PriorProductionTraining",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        logger('Starting production training...', "INFO")
        
        results = run_prior_production_training(
            spark=spark,
            db_name=DB_NAME,
            collection_prefix=COLLECTION_PREFIX,
            collection_suffix=COLLECTION_SUFFIX,
            device=device,
            codebook_size=codebook_size,
            best_config=best_config,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            production_dir=PRODUCTION_DIR
        )
        
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('PRODUCTION TRAINING COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Models trained: {results["num_splits"]}', "INFO")
        logger(f'Avg validation loss: {results["avg_val_loss"]:.4f}', "INFO")
        logger(f'Models saved to: {PRODUCTION_DIR}', "INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if not is_orchestrated:
            spark.stop()


if __name__ == "__main__":
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    
    import time
    start_time = time.time()
    
    try:
        main()
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        logger('', "INFO")
        logger(f'Total time: {hours}h {minutes}m', "INFO")
        logger('Stage 14b completed successfully', "INFO")
        
    except Exception:
        sys.exit(1)