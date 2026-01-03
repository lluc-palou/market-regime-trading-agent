"""
Stage 09: Apply Feature Transformations

TRAIN MODE: Applies transformations to split_X_input collections using per-split fitted params
TEST MODE: Applies transformations to test_data using fitted params from split_0

Usage:
    TRAIN: python scripts/08_apply_feature_transforms.py --mode train
    TEST:  python scripts/08_apply_feature_transforms.py --mode test --test-split 0
"""

import os
import sys
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import json
import mlflow

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.utils.database import write_to_mongodb
from src.feature_transformation import identify_feature_names

# Import the per-split cyclic manager
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))
from src.split_materialization.per_split_cyclic_manager import PerSplitCyclicManager

# Import direct transformation approach (simpler than UDFs)
from src.feature_transformation.transformation_application import apply_transformations_direct

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"

# Input/Output collections (simplified pattern)
INPUT_COLLECTION_TEMPLATE = "split_{}_input"
OUTPUT_COLLECTION_TEMPLATE = "split_{}_output"

# Artifacts directory (same level as src/)
REPO_ROOT = Path(SCRIPT_DIR).parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "feature_transformation"

# MLflow configuration (for reference, but we read from artifacts/)
MLFLOW_TRACKING_URI = "mlruns/"
MLFLOW_EXPERIMENT_NAME = "Feature_Transformation"

# Spark configuration
MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "4g"

# Processing configuration
MAX_SPLITS = None  # None = all splits, or set to limit
FORCE_OVERWRITE = True  # Overwrite existing split_X_working collections

# =================================================================================================
# Helper Functions
# =================================================================================================

def load_final_transforms() -> dict:
    """
    Load final transformation selections from artifacts directory.
    
    Returns:
        Dictionary mapping feature_name -> transform_type
    """
    logger("Loading transformation selections from artifacts...", "INFO")
    
    final_transforms_path = ARTIFACTS_DIR / "aggregation" / "final_transforms.json"
    
    if not final_transforms_path.exists():
        raise FileNotFoundError(
            f"final_transforms.json not found at {final_transforms_path}\n"
            f"Please run Stage 7 first: python scripts/07_select_feature_transforms.py"
        )
    
    with open(final_transforms_path, 'r') as f:
        final_transforms = json.load(f)
    
    logger(f"Loaded transformations for {len(final_transforms)} features", "INFO")
    logger(f"From: {final_transforms_path}", "INFO")
    
    return final_transforms


def load_split_fitted_params(split_id: int) -> dict:
    """
    Load fitted transformation parameters for a specific split from artifacts.

    Args:
        split_id: Split ID

    Returns:
        Dictionary with fitted parameters per feature per transform
    """
    logger(f"Loading fitted parameters for split {split_id}...", "INFO")

    fitted_params_path = ARTIFACTS_DIR / f"split_{split_id}" / f"split_{split_id}_fitted_params.json"

    if not fitted_params_path.exists():
        raise FileNotFoundError(
            f"Fitted parameters not found at {fitted_params_path}\n"
            f"Please run Stage 7 first: python scripts/07_select_feature_transforms.py"
        )

    with open(fitted_params_path, 'r') as f:
        fitted_params = json.load(f)

    logger(f"Loaded fitted parameters for {len(fitted_params)} features", "INFO")
    logger(f"From: {fitted_params_path}", "INFO")

    return fitted_params


def load_test_mode_artifacts() -> tuple:
    """
    Load test mode transformation selections and fitted parameters from artifacts.

    Returns:
        Tuple of (final_transforms, fitted_params)
    """
    logger("Loading test mode artifacts...", "INFO")

    test_mode_dir = ARTIFACTS_DIR / "test_mode"

    # Load final transforms
    transforms_path = test_mode_dir / "final_transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Test mode transforms not found at {transforms_path}\n"
            f"Please run Stage 7 in test mode first: python scripts/07_feature_transform.py --mode test"
        )

    with open(transforms_path, 'r') as f:
        final_transforms = json.load(f)

    logger(f"Loaded transformations for {len(final_transforms)} features", "INFO")
    logger(f"From: {transforms_path}", "INFO")

    # Load fitted params
    params_path = test_mode_dir / "fitted_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Test mode fitted params not found at {params_path}\n"
            f"Please run Stage 7 in test mode first: python scripts/07_feature_transform.py --mode test"
        )

    with open(params_path, 'r') as f:
        fitted_params = json.load(f)

    logger(f"Loaded fitted parameters for {len(fitted_params)} features", "INFO")
    logger(f"From: {params_path}", "INFO")

    return final_transforms, fitted_params


# =================================================================================================
# Transformation Application (Placeholder - to be implemented)
# =================================================================================================

def apply_transformations_to_split(spark, split_id: int, 
                                  final_transforms: dict, 
                                  fitted_params: dict):
    """
    Apply transformations to a split.
    
    Processing:
    - Read from split_X_input
    - Identify actual features from data (using feature_names array)
    - Apply selected transformations with fitted parameters
    - Write to split_X_output
    
    Args:
        spark: SparkSession
        split_id: Split ID to process
        final_transforms: Selected transformation types per feature
        fitted_params: Fitted parameters per feature
    """
    logger(f"=" * 80, "INFO")
    logger(f"APPLYING TRANSFORMATIONS TO SPLIT {split_id}", "INFO")
    logger(f"=" * 80, "INFO")
    
    input_coll = INPUT_COLLECTION_TEMPLATE.format(split_id)
    output_coll = OUTPUT_COLLECTION_TEMPLATE.format(split_id)
    
    logger(f"Reading from: {input_coll}", "INFO")
    logger(f"Writing to: {output_coll}", "INFO")
    
    # Load data from input collection
    logger("Loading data from MongoDB...", "INFO")
    df = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", input_coll)
        .load()
    )
    
    original_count = df.count()
    logger(f"Loaded {original_count:,} documents", "INFO")
    
    # Identify features from data (same way as Stage 7)
    logger("Identifying features from data...", "INFO")
    feature_names_from_data = identify_feature_names(df)
    logger(f"Found {len(feature_names_from_data)} features in data", "INFO")
    
    # Diagnostic: Compare with requested features
    available_features = set(feature_names_from_data)
    requested_features = set(final_transforms.keys())
    missing_features = requested_features - available_features
    extra_features = available_features - requested_features
    
    if missing_features:
        logger("", "INFO")
        logger("=" * 80, "WARNING")
        logger(f"FEATURE MISMATCH DETECTED", "WARNING")
        logger("=" * 80, "WARNING")
        logger(f"Requested features: {len(requested_features)}", "WARNING")
        logger(f"Available features: {len(available_features)}", "WARNING")
        logger(f"Missing features: {len(missing_features)}", "WARNING")
        if len(missing_features) <= 10:
            logger("Missing features list:", "WARNING")
            for feat in sorted(missing_features):
                logger(f"  - {feat} (requested transform: {final_transforms[feat]})", "WARNING")
        else:
            logger(f"Too many missing features to list ({len(missing_features)})", "WARNING")
        logger("=" * 80, "WARNING")
        logger("", "INFO")
    
    if extra_features:
        logger(f"Note: {len(extra_features)} features in data but not in transformation config (will be kept as-is)", "INFO")
    
    # Filter final_transforms to only include available features
    transforms_to_apply = {
        feat: transform 
        for feat, transform in final_transforms.items() 
        if feat in available_features
    }
    
    logger(f"Will apply transformations to {len(transforms_to_apply)} features", "INFO")
    
    # Use direct transformation with Stage 7's proven data loading functions
    logger("=" * 80, "INFO")
    logger("APPLYING TRANSFORMATIONS (Using Stage 7's data loading)", "INFO")
    logger("=" * 80, "INFO")
    
    apply_transformations_direct(
        spark=spark,
        db_name=DB_NAME,
        input_collection=input_coll,
        output_collection=output_coll,
        feature_names=feature_names_from_data,
        final_transforms=transforms_to_apply,
        fitted_params=fitted_params
    )
    
    logger(f"Successfully completed transformation for split {split_id}", "INFO")


def apply_transformations_to_test_data(spark, test_split: int,
                                        final_transforms: dict,
                                        fitted_params: dict):
    """
    Apply transformations to test_data collection using fitted params from test_split.

    Processing:
    - Read from test_data collection
    - Apply transformations using fitted params from specified split
    - Write to test_data_transformed
    - Swap test_data_transformed -> test_data

    Args:
        spark: SparkSession
        test_split: Split ID to use for fitted parameters (typically 0)
        final_transforms: Selected transformation types per feature
        fitted_params: Fitted parameters from test_split
    """
    logger(f"=" * 80, "INFO")
    logger(f"APPLYING TRANSFORMATIONS TO TEST_DATA", "INFO")
    logger(f"Using fitted parameters from split_{test_split}", "INFO")
    logger(f"=" * 80, "INFO")

    input_coll = 'test_data'
    output_coll = 'test_data_transformed'

    logger(f"Reading from: {input_coll}", "INFO")
    logger(f"Writing to: {output_coll}", "INFO")

    # Load data from test_data collection
    logger("Loading data from MongoDB...", "INFO")
    df = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", input_coll)
        .load()
    )

    original_count = df.count()
    logger(f"Loaded {original_count:,} documents", "INFO")

    # Identify features from data
    logger("Identifying features from data...", "INFO")
    feature_names_from_data = identify_feature_names(df)
    logger(f"Found {len(feature_names_from_data)} features in data", "INFO")

    # Filter transforms to only include available features
    available_features = set(feature_names_from_data)
    transforms_to_apply = {
        feat: transform
        for feat, transform in final_transforms.items()
        if feat in available_features
    }

    logger(f"Will apply transformations to {len(transforms_to_apply)} features", "INFO")

    # Apply transformations
    logger("=" * 80, "INFO")
    logger("APPLYING TRANSFORMATIONS", "INFO")
    logger("=" * 80, "INFO")

    apply_transformations_direct(
        spark=spark,
        db_name=DB_NAME,
        input_collection=input_coll,
        output_collection=output_coll,
        feature_names=feature_names_from_data,
        final_transforms=transforms_to_apply,
        fitted_params=fitted_params
    )

    # Swap: test_data_transformed -> test_data
    logger("", "INFO")
    logger("Swapping transformed data to test_data...", "INFO")

    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    # Drop original test_data
    db[input_coll].drop()
    logger(f"  Dropped {input_coll}", "INFO")

    # Rename output to test_data
    db[output_coll].rename(input_coll)
    logger(f"  Renamed {output_coll} → {input_coll}", "INFO")

    client.close()

    logger(f"Successfully completed transformation for test_data", "INFO")


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply feature transformations')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Pipeline mode: train (transform splits) or test (transform test_data)')
    parser.add_argument('--test-split', type=int, default=0,
                        help='Split ID to use for fitted params in test mode (default: 0)')
    args = parser.parse_args()

    mode = args.mode
    test_split = args.test_split

    logger('=' * 80, "INFO")
    logger(f'STAGE 09: APPLY FEATURE TRANSFORMATIONS - {mode.upper()} MODE', "INFO")
    logger('=' * 80, "INFO")
    
    if mode == 'train':
        # TRAIN MODE: Process all splits
        manager = PerSplitCyclicManager(MONGO_URI, DB_NAME)

        try:
            # Discover splits
            split_ids = manager.get_split_ids()

            if not split_ids:
                logger("No split_X_input collections found!", "ERROR")
                return 1

            logger(f"Found {len(split_ids)} splits: {split_ids}", "INFO")

            # Apply max splits limit
            if MAX_SPLITS is not None:
                split_ids = split_ids[:MAX_SPLITS]
                logger(f"Processing first {MAX_SPLITS} splits", "INFO")

            # Show initial state
            logger("", "INFO")
            manager.print_all_splits_state()

            # Create timestamp indexes
            logger("", "INFO")
            logger("Creating timestamp indexes on all split collections...", "INFO")
            from pymongo import MongoClient, ASCENDING
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]

            for split_id in split_ids:
                input_collection = INPUT_COLLECTION_TEMPLATE.format(split_id)
                input_coll = db[input_collection]

                existing_indexes = list(input_coll.list_indexes())
                has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

                if not has_timestamp_index:
                    logger(f'  Creating index on {input_collection}...', "INFO")
                    input_coll.create_index([("timestamp", ASCENDING)], background=False)
                else:
                    logger(f'  Index already exists on {input_collection}', "INFO")

            client.close()
            logger('Timestamp indexes created/verified on all split collections', "INFO")
            logger('', "INFO")

            # Load transformation selections
            final_transforms = load_final_transforms()

            # Create Spark session
            logger("", "INFO")
            logger("Initializing Spark...", "INFO")
            spark = create_spark_session(
                app_name="ApplyFeatureTransforms",
                db_name=DB_NAME,
                mongo_uri=MONGO_URI
            )

            try:
                # Process each split
                for split_id in split_ids:
                    logger("", "INFO")
                    logger("=" * 80, "INFO")
                    logger(f"PROCESSING SPLIT {split_id}", "INFO")
                    logger("=" * 80, "INFO")

                    # Validate input exists
                    if not manager.validate_split_input_exists(split_id):
                        logger(f"Skipping split {split_id} (invalid input)", "ERROR")
                        continue

                    # Prepare for processing (clear output collection)
                    manager.prepare_split_for_processing(split_id, force=FORCE_OVERWRITE)

                    # Load fitted parameters for this split
                    fitted_params = load_split_fitted_params(split_id)

                    # Apply transformations: split_X_input -> split_X_output
                    apply_transformations_to_split(
                        spark,
                        split_id,
                        final_transforms,
                        fitted_params
                    )

                    # Swap: split_X_output -> split_X_input
                    logger("", "INFO")
                    manager.swap_split_to_input(split_id)

                    logger(f"Split {split_id} complete!", "INFO")

                # Show final state
                logger("", "INFO")
                logger("=" * 80, "INFO")
                logger("FINAL STATE", "INFO")
                logger("=" * 80, "INFO")
                manager.print_all_splits_state()

                logger("", "INFO")
                logger("=" * 80, "INFO")
                logger("STAGE 09 COMPLETE (TRAIN MODE)", "INFO")
                logger("=" * 80, "INFO")
                logger(f"Transformed {len(split_ids)} splits", "INFO")
                logger(f"Transformed data is now in split_X_input collections", "INFO")

                return 0

            finally:
                if not is_orchestrated:
                    spark.stop()
                    logger('Spark session stopped', "INFO")

        finally:
            manager.close()

    else:  # TEST MODE
        # TEST MODE: Apply transformations to test_data using artifacts fitted by Stage 7 test mode
        logger("Loading transformations and fitted params from test_mode/ artifacts", "INFO")
        logger("These were fitted by Stage 7 in test mode on split_0", "INFO")
        logger("", "INFO")

        # Create timestamp index on test_data
        logger("Creating timestamp index on test_data collection...", "INFO")
        from pymongo import MongoClient, ASCENDING
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        test_coll = db['test_data']
        existing_indexes = list(test_coll.list_indexes())
        has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

        if not has_timestamp_index:
            logger('  Creating index on test_data...', "INFO")
            test_coll.create_index([("timestamp", ASCENDING)], background=False)
        else:
            logger('  Index already exists on test_data', "INFO")

        client.close()
        logger('Timestamp index created/verified', "INFO")
        logger('', "INFO")

        # Load transformation selections and fitted params from test_mode artifacts
        final_transforms, fitted_params = load_test_mode_artifacts()

        # Create Spark session
        logger("Initializing Spark...", "INFO")
        spark = create_spark_session(
            app_name="ApplyFeatureTransforms_Test",
            db_name=DB_NAME,
            mongo_uri=MONGO_URI
        )

        try:
            # Apply transformations to test_data
            apply_transformations_to_test_data(
                spark,
                test_split,
                final_transforms,
                fitted_params
            )

            logger("", "INFO")
            logger("=" * 80, "INFO")
            logger("STAGE 09 COMPLETE (TEST MODE)", "INFO")
            logger("=" * 80, "INFO")
            logger("Transformed test_data using test_mode artifacts", "INFO")
            logger("Transformed data is now in test_data collection", "INFO")

            return 0

        finally:
            if not is_orchestrated:
                spark.stop()
                logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    # Checks if runned from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    sys.exit(main())