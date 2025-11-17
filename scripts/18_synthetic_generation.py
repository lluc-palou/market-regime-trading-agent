"""
Synthetic LOB Generation Script (Stage 15)

Generates synthetic LOB sequences using trained prior and VQ-VAE models.

Input: split_X_prior.pth + split_X_vqvae_model.pth
Output: split_X_synthetic collections

Usage:
    python scripts/15_synthetic_lob_generation.py
"""

import os
import sys
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
from datetime import datetime

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.vqvae_representation.data_loader import discover_splits
from src.prior.synthetic_generator import SyntheticLOBGenerator, load_models_for_generation
from src.prior.prior_config import GENERATION_CONFIG

# Configuration
DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LOB_Synthetic_Generation"

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "8g"

PRIOR_MODEL_DIR = Path(REPO_ROOT) / "artifacts" / "prior_models" / "production"
VQVAE_MODEL_DIR = Path(REPO_ROOT) / "artifacts" / "vqvae_models" / "production"


def main():
    logger('=' * 100, "INFO")
    logger('SYNTHETIC LOB GENERATION (STAGE 15)', "INFO")
    logger('=' * 100, "INFO")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger(f'Device: {device}', "INFO")
    
    if device.type == 'cuda':
        logger(f'CUDA Device: {torch.cuda.get_device_name(0)}', "INFO")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark (for discovering splits)
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="SyntheticLOBGeneration",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Discover splits
        split_ids = discover_splits(spark, DB_NAME, COLLECTION_PREFIX, COLLECTION_SUFFIX)
        
        if not split_ids:
            raise ValueError("No splits found")
        
        logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")
        logger('', "INFO")
        logger('Generation configuration:', "INFO")
        for key, value in GENERATION_CONFIG.items():
            logger(f'  {key}: {value}', "INFO")
        
        all_results = []
        
        with mlflow.start_run(run_name=f"synthetic_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("db_name", DB_NAME)
            mlflow.log_param("num_splits", len(split_ids))
            for key, value in GENERATION_CONFIG.items():
                mlflow.log_param(key, value)
            
            # Process each split
            for split_id in split_ids:
                logger('', "INFO")
                logger('=' * 100, "INFO")
                logger(f'SPLIT {split_id}', "INFO")
                logger('=' * 100, "INFO")
                
                # Model paths
                prior_model_path = PRIOR_MODEL_DIR / f"split_{split_id}_prior.pth"
                vqvae_model_path = VQVAE_MODEL_DIR / f"split_{split_id}_model.pth"
                
                if not prior_model_path.exists():
                    logger(f'Prior model not found: {prior_model_path}', "WARNING")
                    continue
                
                if not vqvae_model_path.exists():
                    logger(f'VQ-VAE model not found: {vqvae_model_path}', "WARNING")
                    continue
                
                # Load models
                logger(f'Loading models...', "INFO")
                prior_model, vqvae_model = load_models_for_generation(
                    str(prior_model_path),
                    str(vqvae_model_path),
                    device
                )
                
                # Create generator
                generator = SyntheticLOBGenerator(
                    prior_model=prior_model,
                    vqvae_model=vqvae_model,
                    device=device,
                    mongo_uri=MONGO_URI
                )
                
                # Generate
                output_collection = f"{COLLECTION_PREFIX}{split_id}_synthetic"
                
                with mlflow.start_run(
                    run_name=f"split_{split_id}_generation",
                    nested=True
                ):
                    result = generator.generate_and_save(
                        db_name=DB_NAME,
                        output_collection=output_collection,
                        split_id=split_id
                    )
                    
                    # Log to MLflow
                    mlflow.log_param("split_id", split_id)
                    mlflow.log_param("output_collection", output_collection)
                    mlflow.log_metric("total_sequences", result['total_sequences'])
                    mlflow.log_metric("total_samples", result['total_samples'])
                    mlflow.log_metric("seq_len", result['seq_len'])
                    mlflow.log_metric("temperature", result['temperature'])
                    
                    all_results.append({
                        'split_id': split_id,
                        'output_collection': output_collection,
                        **result
                    })
                
                generator.close()
                
                logger(f'Synthetic data saved to: {output_collection}', "INFO")
            
            # Aggregate metrics
            total_sequences = sum(r['total_sequences'] for r in all_results)
            total_samples = sum(r['total_samples'] for r in all_results)
            
            mlflow.log_metric("total_sequences_all_splits", total_sequences)
            mlflow.log_metric("total_samples_all_splits", total_samples)
        
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('GENERATION COMPLETE', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Splits processed: {len(all_results)}', "INFO")
        logger(f'Total sequences: {total_sequences}', "INFO")
        logger(f'Total samples: {total_samples}', "INFO")
        
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
        logger('Stage 15 completed successfully', "INFO")
        
    except Exception:
        sys.exit(1)