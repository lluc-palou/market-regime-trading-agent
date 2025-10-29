"""
EWMA Half-Life Selection Script

Selects optimal EWMA half-life parameters for LOB feature standardization using CPCV splits.

This is Stage 10 in the pipeline - it follows feature transformation (Stage 8).

Usage:
    python scripts/10_select_ewma_halflife.py
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
from src.feature_standardization import (
    EWMAHalfLifeProcessor,
    aggregate_across_splits,
    select_final_half_lives,
    identify_feature_names,
    filter_standardizable_features
)
from src.feature_standardization.mlflow_logger import (
    log_split_results,
    log_aggregated_results
)

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
INPUT_COLLECTION_PREFIX = "split_"
INPUT_COLLECTION_SUFFIX = "_input"  # Read from transformation output (renamed to _input)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LOB_EWMA_HalfLife_Selection"

MAX_SPLITS = 5
TRAIN_SAMPLE_RATE = 0.1  # 10% sampling for efficiency
CLIP_STD = 3.0

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/Users/llucp/spark_jars/"
DRIVER_MEMORY = "4g"

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 80, "INFO")
    logger('EWMA HALF-LIFE SELECTION', "INFO")
    logger('=' * 80, "INFO")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
    
    # Create Spark session
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="EWMAHalfLifeSelection",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory=DRIVER_MEMORY,
        jar_files_path=JAR_FILES_PATH
    )
    
    try:
        # Get feature names from first split
        logger('Identifying features...', "INFO")
        first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
        logger(f'Loading from collection: {first_split}', "INFO")
        
        # Load one document to get feature_names
        sample_df = (
            spark.read.format("mongodb")
            .option("database", DB_NAME)
            .option("collection", first_split)
            .load()
            .limit(1)
        )
        
        # Check if collection has data
        count = sample_df.count()
        if count == 0:
            raise ValueError(f"Collection '{first_split}' is empty! Ensure Stage 8 (transformation) completed successfully.")
        
        all_feature_names = identify_feature_names(sample_df)
        
        # Don't filter yet - we need all features to match the features array indices
        # We'll filter which ones to standardize during processing
        feature_names = all_feature_names
        standardizable_features = filter_standardizable_features(all_feature_names)
        
        logger(f'Processing {len(feature_names)} total features', "INFO")
        logger(f'Will standardize {len(standardizable_features)} features '
               f'(excluded {len(feature_names) - len(standardizable_features)})', "INFO")
        
        # Initialize processor
        processor = EWMAHalfLifeProcessor(
            spark=spark,
            db_name=DB_NAME,
            input_collection_prefix=INPUT_COLLECTION_PREFIX,
            input_collection_suffix=INPUT_COLLECTION_SUFFIX,
            train_sample_rate=TRAIN_SAMPLE_RATE,
            clip_std=CLIP_STD
        )
        
        # Process each split
        all_split_results = {}
        
        for split_id in range(MAX_SPLITS):
            # Process split - pass both full feature names and standardizable subset
            split_results = processor.process_split(split_id, feature_names, standardizable_features)
            all_split_results[split_id] = split_results
            
            # Log to MLflow
            log_split_results(split_id, split_results, TRAIN_SAMPLE_RATE)
        
        # Aggregate across splits
        logger('', "INFO")  # Blank line
        aggregated = aggregate_across_splits(all_split_results)
        
        # Select final half-lives
        final_half_lives = select_final_half_lives(aggregated, strategy='most_frequent')
        
        # Log aggregated results
        log_aggregated_results(aggregated, final_half_lives)
        
        # Summary
        logger('', "INFO")
        logger('=' * 80, "INFO")
        logger('HALF-LIFE SELECTION COMPLETE', "INFO")
        logger('=' * 80, "INFO")
        logger(f'Results logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")
        logger(f'Final half-lives saved to: artifacts/ewma_halflife_selection/aggregation/final_halflifes.json', "INFO")
        
        # Show sample of selected half-lives
        logger('', "INFO")
        logger('Sample of selected half-lives:', "INFO")
        sample_features = list(final_half_lives.items())[:10]
        for feat, hl in sample_features:
            logger(f'  {feat}: {hl}', "INFO")
        if len(final_half_lives) > 10:
            logger(f'  ... and {len(final_half_lives) - 10} more', "INFO")
        
    finally:
        spark.stop()
        logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    main()