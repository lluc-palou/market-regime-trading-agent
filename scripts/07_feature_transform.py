"""
Feature Transformation Selection Script

Selects optimal normalization transformations for LOB features using CPCV splits.

Usage:
    python scripts/07_select_feature_transforms.py
"""

import os
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import mlflow

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.feature_transformation import (
    FeatureTransformProcessor,
    aggregate_across_splits,
    select_final_transforms,
    identify_feature_names
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
INPUT_COLLECTION_SUFFIX = "_input"  # Read from split_X_input collections

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LOB_Feature_Transform_Selection"

MAX_SPLITS = 5
TRAIN_SAMPLE_RATE = 1.0  # 1.0 = all data, 0.1 = 10%

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
DRIVER_MEMORY = "4g"

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 80, "INFO")
    logger('FEATURE TRANSFORMATION SELECTION', "INFO")
    logger('=' * 80, "INFO")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark session
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="FeatureTransformSelection",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Get feature names from first split
        logger('Identifying features...', "INFO")
        first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
        sample_df = (
            spark.read.format("mongodb")
            .option("database", DB_NAME)
            .option("collection", first_split)
            .load()
            .limit(1)
        )
        
        feature_names = identify_feature_names(sample_df)
        logger(f'Processing {len(feature_names)} features across {MAX_SPLITS} splits', "INFO")
        
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
            # Process split
            split_results = processor.process_split(split_id, feature_names)
            all_split_results[split_id] = split_results
            
            # Log to MLflow
            log_split_results(split_id, split_results, TRAIN_SAMPLE_RATE)
        
        # Aggregate across splits
        logger('', "INFO")  # Blank line
        aggregated = aggregate_across_splits(all_split_results)
        
        # Select final transforms
        final_transforms = select_final_transforms(aggregated, strategy='most_frequent')
        
        # Log aggregated results
        log_aggregated_results(aggregated, final_transforms)
        
        # Summary
        logger('', "INFO")
        logger('=' * 80, "INFO")
        logger('TRANSFORMATION SELECTION COMPLETE', "INFO")
        logger('=' * 80, "INFO")
        logger(f'Results logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
        logger(f'Final transforms saved to: final_transforms.json', "INFO")
        
    finally:
        spark.stop()
        logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    main()