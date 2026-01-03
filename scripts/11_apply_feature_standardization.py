"""
Apply EWMA Standardization Script (Stage 12)

TRAIN MODE: Applies EWMA standardization to split_X_input collections
TEST MODE: Applies EWMA standardization to test_data using half-lives from split_0

Usage:
    TRAIN: python scripts/11_apply_feature_standardization.py --mode train
    TEST:  python scripts/11_apply_feature_standardization.py --mode test --test-split 0

    # For parallel execution in train mode:
    python scripts/11_apply_feature_standardization.py --mode train --splits 0,2,4,6,8 &
    python scripts/11_apply_feature_standardization.py --mode train --splits 1,3,5,7,9 &
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.feature_standardization import (
    identify_feature_names,
    filter_standardizable_features
)
from src.feature_standardization.apply_scaler import EWMAStandardizationApplicator

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
INPUT_COLLECTION_PREFIX = "split_"
INPUT_COLLECTION_SUFFIX = "_input"  # Read from transformation output

OUTPUT_COLLECTION_PREFIX = "split_"
OUTPUT_COLLECTION_SUFFIX = "_output"  # Write for next stage (cyclic pattern)

# Path to half-life selection results (relative to repository root)
# Support both JSON (legacy) and CSV formats
HALFLIFE_RESULTS_JSON = Path(REPO_ROOT) / "artifacts" / "ewma_halflife_selection" / "aggregation" / "final_halflifes.json"
HALFLIFE_RESULTS_CSV = Path(REPO_ROOT) / "artifacts" / "ewma_halflife_selection" / "aggregation" / "halflife_frequency.csv"
# Alternative CSV location
HALFLIFE_RESULTS_CSV_ALT = Path(REPO_ROOT) / "artifacts" / "feature_scale" / "halflife_frequency.csv"

# Scaler state artifacts (fitted parameters)
SCALER_STATES_DIR = Path(REPO_ROOT) / "artifacts" / "ewma_standardization" / "scaler_states"
TEST_MODE_SCALER_STATES = SCALER_STATES_DIR / "test_mode_scaler_states.json"

CLIP_STD = 3.0

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "4g"

# =================================================================================================
# Main Execution
# =================================================================================================

def save_scaler_states(applicator: EWMAStandardizationApplicator, filepath: Path):
    """
    Save fitted scaler states to JSON file.

    Args:
        applicator: EWMAStandardizationApplicator with fitted scalers
        filepath: Path to save scaler states
    """
    scaler_states = applicator.get_scaler_states()

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(scaler_states, f, indent=2)

    logger(f'Saved scaler states to: {filepath}', "INFO")
    logger(f'  Features: {len(scaler_states)}', "INFO")


def load_scaler_states(applicator: EWMAStandardizationApplicator, filepath: Path):
    """
    Load fitted scaler states from JSON file and update applicator scalers.

    Args:
        applicator: EWMAStandardizationApplicator to update
        filepath: Path to load scaler states from
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler states file not found: {filepath}")

    # Load from JSON
    with open(filepath, 'r') as f:
        scaler_states = json.load(f)

    # Update each scaler with loaded state
    for feat_name, state in scaler_states.items():
        if feat_name in applicator.scalers:
            scaler = applicator.scalers[feat_name]
            scaler.ewma_mean = state['ewma_mean']
            scaler.ewma_var = state['ewma_var']
            scaler.initialized = True
            scaler.n_samples = state['n_samples']

    logger(f'Loaded scaler states from: {filepath}', "INFO")
    logger(f'  Features: {len(scaler_states)}', "INFO")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply EWMA standardization to collections')
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        default='train',
        help='Pipeline mode: train (standardize splits) or test (standardize test_data)'
    )
    parser.add_argument(
        '--test-split',
        type=int,
        default=0,
        help='Split ID to use for half-lives in test mode (default: 0)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        default=None,
        help='[TRAIN MODE ONLY] Comma-separated list of split IDs to process (e.g., "0,1,2,3"). If not specified, processes all splits.'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()

    mode = args.mode
    test_split = args.test_split

    logger('=' * 80, "INFO")
    logger(f'APPLY EWMA STANDARDIZATION (STAGE 12) - {mode.upper()} MODE', "INFO")
    logger('=' * 80, "INFO")

    if mode == 'train':
        if args.splits:
            requested_splits = [int(s.strip()) for s in args.splits.split(',')]
            logger(f'Processing specific splits: {requested_splits}', "INFO")
        else:
            requested_splits = None
            logger('Processing all available splits', "INFO")
    else:  # test mode
        logger(f'Processing test_data using half-lives from split_{test_split}', "INFO")
        requested_splits = None
    
    # Load half-life results from Stage 10
    # Try CSV first, then JSON
    halflife_path = None
    if HALFLIFE_RESULTS_CSV.exists():
        halflife_path = HALFLIFE_RESULTS_CSV
        logger(f'Loading half-lives from CSV: {halflife_path}', "INFO")
    elif HALFLIFE_RESULTS_CSV_ALT.exists():
        halflife_path = HALFLIFE_RESULTS_CSV_ALT
        logger(f'Loading half-lives from CSV: {halflife_path}', "INFO")
    elif HALFLIFE_RESULTS_JSON.exists():
        halflife_path = HALFLIFE_RESULTS_JSON
        logger(f'Loading half-lives from JSON: {halflife_path}', "INFO")
    else:
        logger(f'ERROR: Half-life results not found!', "ERROR")
        logger(f'Checked locations:', "ERROR")
        logger(f'  - {HALFLIFE_RESULTS_CSV}', "ERROR")
        logger(f'  - {HALFLIFE_RESULTS_CSV_ALT}', "ERROR")
        logger(f'  - {HALFLIFE_RESULTS_JSON}', "ERROR")
        logger('Please run Stage 10 (10_feature_scale.py) first', "ERROR")
        return

    # Load based on file format
    if halflife_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(halflife_path)

        # Detect column names
        if 'feature' in df.columns and 'half_life' in df.columns:
            feature_col, halflife_col = 'feature', 'half_life'
        elif 'feature' in df.columns and 'halflife' in df.columns:
            feature_col, halflife_col = 'feature', 'halflife'
        elif 'feature_name' in df.columns and 'half_life' in df.columns:
            feature_col, halflife_col = 'feature_name', 'half_life'
        else:
            # Assume first two columns
            feature_col, halflife_col = df.columns[0], df.columns[1]
            logger(f'Using columns: {feature_col} (feature), {halflife_col} (half_life)', "INFO")

        # Convert to dictionary
        final_halflifes = {}
        for _, row in df.iterrows():
            feature = row[feature_col]
            halflife = int(row[halflife_col])  # Convert to int
            final_halflifes[feature] = halflife
    else:  # JSON
        with open(halflife_path, 'r') as f:
            final_halflifes = json.load(f)

    logger(f'Loaded half-lives for {len(final_halflifes)} features', "INFO")
    
    # Show sample half-lives
    sample_halflifes = list(final_halflifes.items())[:5]
    for feat, hl in sample_halflifes:
        logger(f'  {feat}: half_life={hl}', "INFO")
    if len(final_halflifes) > 5:
        logger(f'  ... and {len(final_halflifes) - 5} more', "INFO")
    
    # Create Spark session
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name=f"EWMAStandardization_{mode.title()}",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI
    )

    try:
        if mode == 'train':
            # TRAIN MODE: Process splits
            # Get feature names from first split
            logger('Identifying features...', "INFO")
            first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
            logger(f'Loading from collection: {first_split}', "INFO")

            sample_df = (
                spark.read.format("mongodb")
                .option("database", DB_NAME)
                .option("collection", first_split)
                .load()
                .limit(1)
            )

            count = sample_df.count()
            if count == 0:
                raise ValueError(f"Collection '{first_split}' is empty!")

            all_feature_names = identify_feature_names(sample_df)

            logger(f'Total features: {len(all_feature_names)}', "INFO")
            logger(f'Standardizing: {len(final_halflifes)} features', "INFO")

            # Discover all split collections
            from pymongo import MongoClient
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]
            all_collections = db.list_collection_names()

            # Extract split IDs from collection names matching pattern
            import re
            split_pattern = re.compile(rf'^{INPUT_COLLECTION_PREFIX}(\d+){INPUT_COLLECTION_SUFFIX}$')
            split_ids = []
            for coll_name in all_collections:
                match = split_pattern.match(coll_name)
                if match:
                    split_ids.append(int(match.group(1)))
            split_ids = sorted(split_ids)
            client.close()

            if not split_ids:
                raise ValueError(f'No split collections found matching pattern: {INPUT_COLLECTION_PREFIX}X{INPUT_COLLECTION_SUFFIX}')

            logger(f'Found {len(split_ids)} split collections: {split_ids}', "INFO")

            # Filter to requested splits if specified
            if requested_splits is not None:
                split_ids = [sid for sid in split_ids if sid in requested_splits]
                if not split_ids:
                    raise ValueError(f'None of the requested splits {requested_splits} were found in database')
                logger(f'Filtered to requested splits: {split_ids}', "INFO")
            else:
                logger(f'Processing all {len(split_ids)} splits', "INFO")

            # Create timestamp indexes
            logger('', "INFO")
            logger('Creating timestamp indexes on all split collections...', "INFO")
            from pymongo import ASCENDING
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]

            for split_id in split_ids:
                input_collection = f"{INPUT_COLLECTION_PREFIX}{split_id}{INPUT_COLLECTION_SUFFIX}"
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

            # Initialize applicator
            applicator = EWMAStandardizationApplicator(
                spark=spark,
                db_name=DB_NAME,
                final_halflifes=final_halflifes,
                clip_std=CLIP_STD
            )

            # Process each split
            for split_id in split_ids:
                logger('', "INFO")

                # Apply standardization
                total_processed = applicator.apply_to_split(
                    split_id=split_id,
                    feature_names=all_feature_names,
                    input_collection_prefix=INPUT_COLLECTION_PREFIX,
                    input_collection_suffix=INPUT_COLLECTION_SUFFIX,
                    output_collection_prefix=OUTPUT_COLLECTION_PREFIX,
                    output_collection_suffix=OUTPUT_COLLECTION_SUFFIX
                )

                # Reset scalers for next split
                applicator.scalers = {
                    feat: type(scaler)(scaler.half_life)
                    for feat, scaler in applicator.scalers.items()
                }

            # Summary
            logger('', "INFO")
            logger('=' * 80, "INFO")
            logger('EWMA STANDARDIZATION COMPLETE (TRAIN MODE)', "INFO")
            logger('=' * 80, "INFO")
            logger(f'Processed {len(split_ids)} splits', "INFO")

            # Rename collections: output -> input
            logger('', "INFO")
            logger('Renaming collections for cyclic pattern...', "INFO")

            from pymongo import MongoClient
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]

            for split_id in split_ids:
                output_collection = f"{OUTPUT_COLLECTION_PREFIX}{split_id}{OUTPUT_COLLECTION_SUFFIX}"
                input_collection = f"{INPUT_COLLECTION_PREFIX}{split_id}{INPUT_COLLECTION_SUFFIX}"

                # Drop old input collection if exists
                if input_collection in db.list_collection_names():
                    db[input_collection].drop()
                    logger(f'  Dropped old {input_collection}', "INFO")

                # Rename output -> input
                db[output_collection].rename(input_collection)
                logger(f'  Renamed {output_collection} -> {input_collection}', "INFO")

            client.close()

            logger('', "INFO")
            logger('Collection renaming complete', "INFO")
            if split_ids:
                logger(f'Next stage will read from: {INPUT_COLLECTION_PREFIX}{{0-{split_ids[-1]}}}{INPUT_COLLECTION_SUFFIX}', "INFO")

        else:  # TEST MODE
            # TEST MODE: Apply standardization to test_data
            logger('Identifying features from test_data...', "INFO")

            sample_df = (
                spark.read.format("mongodb")
                .option("database", DB_NAME)
                .option("collection", "test_data")
                .load()
                .limit(1)
            )

            count = sample_df.count()
            if count == 0:
                raise ValueError("test_data collection is empty!")

            all_feature_names = identify_feature_names(sample_df)

            logger(f'Total features: {len(all_feature_names)}', "INFO")
            logger(f'Standardizing: {len(final_halflifes)} features', "INFO")

            # Create timestamp index on test_data
            logger('', "INFO")
            logger('Creating timestamp index on test_data collection...', "INFO")
            from pymongo import ASCENDING, MongoClient
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

            # ==============================================================================
            # LOAD FITTED SCALERS and APPLY to test_data
            # ==============================================================================
            # NOTE: Scaler states must be fitted by Stage 10 (test mode) first!
            logger('=' * 80, "INFO")
            logger('APPLYING FITTED SCALERS TO TEST_DATA', "INFO")
            logger('=' * 80, "INFO")

            # Check if scaler states exist (fitted by Stage 10)
            if not TEST_MODE_SCALER_STATES.exists():
                logger('', "ERROR")
                logger('ERROR: Scaler states not found!', "ERROR")
                logger(f'Expected: {TEST_MODE_SCALER_STATES}', "ERROR")
                logger('', "ERROR")
                logger('You must run Stage 10 in test mode FIRST to fit scalers:', "ERROR")
                logger('  python scripts/10_ewma_halflife_selection.py --mode test', "ERROR")
                logger('', "ERROR")
                raise FileNotFoundError(f"Scaler states not found: {TEST_MODE_SCALER_STATES}")

            logger(f'Input: test_data', "INFO")
            logger(f'Output: test_data_standardized', "INFO")
            logger(f'Loading scaler states from: {TEST_MODE_SCALER_STATES}', "INFO")
            logger('', "INFO")

            # Initialize fresh applicator for test_data
            applicator_test = EWMAStandardizationApplicator(
                spark=spark,
                db_name=DB_NAME,
                final_halflifes=final_halflifes,
                clip_std=CLIP_STD
            )

            # Load fitted scaler states
            load_scaler_states(applicator_test, TEST_MODE_SCALER_STATES)
            logger('', "INFO")

            # Apply to test_data (transform only, no fitting)
            total_processed = applicator_test.apply_to_collection(
                input_collection='test_data',
                output_collection='test_data_standardized',
                feature_names=all_feature_names
            )

            # Swap: test_data_standardized -> test_data
            logger('', "INFO")
            logger('Swapping standardized data to test_data...', "INFO")

            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]

            # Drop original test_data
            db['test_data'].drop()
            logger('  Dropped test_data', "INFO")

            # Rename output to test_data
            db['test_data_standardized'].rename('test_data')
            logger('  Renamed test_data_standardized → test_data', "INFO")

            client.close()

            # Summary
            logger('', "INFO")
            logger('=' * 80, "INFO")
            logger('EWMA STANDARDIZATION COMPLETE (TEST MODE)', "INFO")
            logger('=' * 80, "INFO")
            logger(f'Processed test_data collection', "INFO")
            logger(f'Standardized data is now in test_data collection', "INFO")
        
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    main()