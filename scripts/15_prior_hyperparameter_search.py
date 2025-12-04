"""
Prior Hyperparameter Search Script

Searches for best prior model hyperparameters across all splits.

Input: split_X_input collections (with latent codes from VQ-VAE)
       VQ-VAE codebook size from best_config.yaml
Output: best_prior_config.json with optimal hyperparameters

Usage:
    python scripts/15_prior_hyperparameter_search.py
"""

import os
import sys
import json
import yaml
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
from src.prior.prior_hyperparameter_search import run_prior_hyperparameter_search

# Configuration
DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LOB_Prior_Hyperparameter_Search"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "8g"

ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts" / "prior_models"

VQVAE_CONFIG_PATH = Path(REPO_ROOT) / "artifacts" / "vqvae_models" / "hyperparameter_search" / "best_config.yaml"


def main():
    logger('=' * 100, "INFO")
    logger('PRIOR HYPERPARAMETER SEARCH', "INFO")
    logger('=' * 100, "INFO")

    # Load VQ-VAE config for codebook size
    if not VQVAE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"VQ-VAE config not found: {VQVAE_CONFIG_PATH}")

    with open(VQVAE_CONFIG_PATH, 'r') as f:
        vqvae_config = yaml.safe_load(f)
        codebook_size = vqvae_config['best_config']['K']

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
        logger('Starting hyperparameter search...', "INFO")

        results = run_prior_hyperparameter_search(
            spark=spark,
            db_name=DB_NAME,
            collection_prefix=COLLECTION_PREFIX,
            collection_suffix=COLLECTION_SUFFIX,
            device=device,
            codebook_size=codebook_size,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            artifact_base_dir=ARTIFACT_BASE_DIR
        )

        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('HYPERPARAMETER SEARCH COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Best configuration: {results["best_config"]}', "INFO")
        logger(f'Best average validation loss: {results["best_avg_val_loss"]:.4f}', "INFO")
        logger(f'Total configurations tested: {len(results["all_results"])}', "INFO")
        
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
        logger('Stage 15 (Prior Hyperparameter Search) completed successfully', "INFO")
        
    except Exception:
        sys.exit(1)