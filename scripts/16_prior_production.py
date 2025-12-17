"""
Prior Production Training Script (Stage 16)

Trains final Prior models using the best hyperparameter configuration from Stage 15.
Saves production-ready Prior models for all splits.

This is Stage 16 in the pipeline - follows Prior hyperparameter search (Stage 15).

Input: split_X_input collections with VQ-VAE latent codes (from Stage 14)
       Best Prior configuration from Stage 15 artifacts
       VQ-VAE codebook size from VQ-VAE best_config.yaml

Output: Trained Prior models saved to artifacts/prior_models/production/split_X/
        Ready for synthetic generation (Stage 17)

Usage:
    python scripts/16_prior_production.py
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
import yaml
import torch
import mlflow
from datetime import datetime

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.prior import run_production_training
from src.prior.prior_config import PRIOR_TRAINING_CONFIG

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"  # Read from VQ-VAE production output (Stage 14)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Prior_Production_Training"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "8g"

# Artifact directories
ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts"
PRIOR_ARTIFACT_DIR = ARTIFACT_BASE_DIR / "prior_models"
VQVAE_ARTIFACT_DIR = ARTIFACT_BASE_DIR / "vqvae_models"

HYPERPARAMETER_SEARCH_DIR = PRIOR_ARTIFACT_DIR / "hyperparameter_search"
PRODUCTION_DIR = PRIOR_ARTIFACT_DIR / "production"

VQVAE_CONFIG_PATH = VQVAE_ARTIFACT_DIR / "hyperparameter_search" / "best_config.yaml"

# =================================================================================================
# Helper Functions
# =================================================================================================

def load_best_prior_config() -> dict:
    """
    Load best Prior hyperparameter configuration from Stage 15.

    Returns:
        Best configuration dictionary
    """
    config_path = HYPERPARAMETER_SEARCH_DIR / "best_prior_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Best Prior config not found at {config_path}. "
            f"Please run Stage 15 (Prior hyperparameter search) first."
        )

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    logger(f'Loaded best Prior config from: {config_path}', "INFO")
    logger(f'  Average validation loss: {config_data["avg_val_loss"]:.4f}', "INFO")
    logger(f'  Configuration: {config_data["config"]}', "INFO")

    return config_data["config"]


def load_vqvae_codebook_size() -> int:
    """
    Load VQ-VAE codebook size from Stage 13 artifacts.

    Returns:
        Codebook size (K)
    """
    if not VQVAE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"VQ-VAE config not found at {VQVAE_CONFIG_PATH}. "
            f"Please run Stages 13-14 (VQ-VAE training) first."
        )

    with open(VQVAE_CONFIG_PATH, 'r') as f:
        vqvae_config = yaml.safe_load(f)

    codebook_size = vqvae_config['best_config']['K']
    logger(f'VQ-VAE codebook size (K): {codebook_size}', "INFO")

    return codebook_size

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 100, "INFO")
    logger('PRIOR PRODUCTION TRAINING (STAGE 16)', "INFO")
    logger('=' * 100, "INFO")

    # Load best Prior configuration
    logger('', "INFO")
    logger('Loading best Prior hyperparameter configuration from Stage 15...', "INFO")
    best_config = load_best_prior_config()

    # Load VQ-VAE codebook size
    logger('', "INFO")
    logger('Loading VQ-VAE configuration...', "INFO")
    codebook_size = load_vqvae_codebook_size()

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
        app_name="PriorProductionTraining",
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
        logger(f'  VQ-VAE codebook size: {codebook_size}', "INFO")

        logger('', "INFO")
        logger('Best Prior hyperparameter configuration:', "INFO")
        for key, value in best_config.items():
            logger(f'  {key}: {value}', "INFO")

        logger('', "INFO")
        logger('Training configuration:', "INFO")
        for key, value in PRIOR_TRAINING_CONFIG.items():
            logger(f'  {key}: {value}', "INFO")

        # CRITICAL: Create timestamp indexes on all split collections for efficient hourly queries
        # Without these indexes, each hourly query performs a full collection scan O(N)
        # With indexes: O(log N + matches) - reduces processing time dramatically
        logger('', "INFO")
        logger('Creating timestamp indexes on all split collections...', "INFO")
        from pymongo import MongoClient, ASCENDING
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        all_collections = db.list_collection_names()

        # Find all split collections
        split_collections = [c for c in all_collections
                           if c.startswith(COLLECTION_PREFIX) and c.endswith(COLLECTION_SUFFIX)]

        for collection_name in split_collections:
            input_coll = db[collection_name]

            # Check if index already exists
            existing_indexes = list(input_coll.list_indexes())
            has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

            if not has_timestamp_index:
                logger(f'  Creating index on {collection_name}...', "INFO")
                input_coll.create_index([("timestamp", ASCENDING)], background=False)
            else:
                logger(f'  Index already exists on {collection_name}', "INFO")

        client.close()
        logger(f'Timestamp indexes created/verified on {len(split_collections)} collections', "INFO")

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
            codebook_size=codebook_size,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            production_dir=PRODUCTION_DIR
        )

        # Summary
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('PRODUCTION TRAINING COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Models trained: {results["num_splits"]}', "INFO")
        logger(f'Average validation loss: {results["avg_val_loss"]:.4f}', "INFO")
        logger(f'Average epochs trained: {results["avg_epochs_trained"]:.1f}', "INFO")
        logger(f'Models saved to: {PRODUCTION_DIR}', "INFO")
        logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")

        logger('', "INFO")
        logger('Next steps:', "INFO")
        logger('  1. Review production models in MLflow UI', "INFO")
        logger('  2. Run Stage 17 (Synthetic LOB generation) using these models', "INFO")
        logger('  3. Each split has its own trained Prior model ready for generation', "INFO")

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
        logger('Stage 16 completed successfully', "INFO")

    except Exception:
        sys.exit(1)
