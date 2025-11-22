"""
Apply EWMA Standardization Script

Applies EWMA standardization using selected half-life parameters from Stage 10.

This is Stage 11 in the pipeline - it follows half-life selection (Stage 10).

Usage:
    python scripts/11_apply_ewma_standardization.py
"""

import os
import sys
import json
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
HALFLIFE_RESULTS_PATH = Path(REPO_ROOT) / "artifacts" / "ewma_halflife_selection" / "aggregation" / "final_halflifes.json"

CLIP_STD = 3.0

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "4g"

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    logger('=' * 80, "INFO")
    logger('APPLY EWMA STANDARDIZATION', "INFO")
    logger('=' * 80, "INFO")
    
    # Load half-life results from Stage 10
    if not HALFLIFE_RESULTS_PATH.exists():
        logger(f'ERROR: Half-life results not found at {HALFLIFE_RESULTS_PATH}', "ERROR")
        logger('Please run Stage 10 (10_select_ewma_halflife.py) first', "ERROR")
        return
    
    with open(HALFLIFE_RESULTS_PATH, 'r') as f:
        final_halflifes = json.load(f)
    
    logger(f'Loaded half-lives for {len(final_halflifes)} features', "INFO")
    
    # Show sample half-lives
    sample_halflifes = list(final_halflifes.items())[:5]
    for feat, hl in sample_halflifes:
        logger(f'  {feat}: half_life={hl}', "INFO")
    if len(final_halflifes) > 5:
        logger(f'  ... and {len(final_halflifes) - 5} more', "INFO")
    
    # Create Spark session (uses default 8GB driver memory and jar path)
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name="EWMAStandardization",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI
    )
    
    try:
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
        client = MongoClient(MONGO_URI)
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

        # CRITICAL: Create timestamp indexes on all split collections for efficient hourly queries
        # Without these indexes, each hourly query performs a full collection scan O(N)
        # With indexes: O(log N + matches) - reduces processing time dramatically
        logger('', "INFO")
        logger('Creating timestamp indexes on all split collections...', "INFO")
        from pymongo import ASCENDING
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        for split_id in split_ids:
            input_collection = f"{INPUT_COLLECTION_PREFIX}{split_id}{INPUT_COLLECTION_SUFFIX}"
            input_coll = db[input_collection]

            # Check if index already exists
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
            logger('', "INFO")  # Blank line
            
            # Apply standardization
            total_processed = applicator.apply_to_split(
                split_id=split_id,
                feature_names=all_feature_names,
                input_collection_prefix=INPUT_COLLECTION_PREFIX,
                input_collection_suffix=INPUT_COLLECTION_SUFFIX,
                output_collection_prefix=OUTPUT_COLLECTION_PREFIX,
                output_collection_suffix=OUTPUT_COLLECTION_SUFFIX
            )
            
            # Reset scalers for next split (each split processes independently)
            applicator.scalers = {
                feat: type(scaler)(scaler.half_life)
                for feat, scaler in applicator.scalers.items()
            }
        
        # Summary
        logger('', "INFO")
        logger('=' * 80, "INFO")
        logger('EWMA STANDARDIZATION COMPLETE', "INFO")
        logger('=' * 80, "INFO")
        logger(f'Processed {len(split_ids)} splits', "INFO")

        # Rename collections: output -> input (cyclic pattern for next stage)
        logger('', "INFO")
        logger('Renaming collections for cyclic pattern...', "INFO")

        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
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
        
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    main()