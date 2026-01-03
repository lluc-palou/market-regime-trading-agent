"""
Feature Transformation Selection Script (Stage 07)

Selects optimal normalization transformations for LOB features using CPCV splits.

TRAIN MODE: Processes all splits, selects best transforms across splits
TEST MODE: Processes split_0 only, saves test_mode artifacts

Input: split_X collections with 18 features
Processes: 16 features (excludes volatility and fwd_logret_1)
Output: Transformation selections for 16 features

Exclusions from transformation:
- volatility: Keep original scale (meaningful interpretation)
- fwd_logret_1: Target variable, keep original scale

Usage:
    TRAIN: python scripts/07_feature_transform.py --mode train
    TEST:  python scripts/07_feature_transform.py --mode test --test-split 0
"""

import os
import sys
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# Unicode/MLflow Fix for Windows
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

# MLflow removed - not needed for this pipeline
# =================================================================================================

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session
from src.feature_transformation import (
    FeatureTransformProcessor,
    aggregate_across_splits,
    select_final_transforms,
    identify_feature_names_from_collection  # FIXED: Use aggregation-based function
)
from src.feature_transformation.transformation_application import fit_selected_transforms_on_full_data

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
INPUT_COLLECTION_PREFIX = "split_"
INPUT_COLLECTION_SUFFIX = "_input"  # Split collections are named split_X_input

TRAIN_SAMPLE_RATE = 0.1

MONGO_URI = "mongodb://127.0.0.1:27017/"
JAR_FILES_PATH = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/"
DRIVER_MEMORY = "4g"

# =================================================================================================
# Feature Filtering
# =================================================================================================

def filter_transformable_features(feature_names):
    """
    Filter features to only include those that should be transformed.
    
    Excludes:
    - volatility: Keep original scale (meaningful interpretation)
    - fwd_logret_*: Target variable, keep original scale
    
    Args:
        feature_names: List of all feature names (18 features from materialization)
        
    Returns:
        List of features to transform (16 features)
    """
    EXCLUDE_PATTERNS = [
        'fwd_logret_',      # Forward returns (targets)
    ]
    
    EXCLUDE_EXACT = []       # Keep original scale
    
    transformable = []
    
    for feat_name in feature_names:
        # Skip exact matches
        if feat_name in EXCLUDE_EXACT:
            continue
        
        # Skip pattern matches
        if any(feat_name.startswith(pattern) for pattern in EXCLUDE_PATTERNS):
            continue
        
        transformable.append(feat_name)
    
    excluded_count = len(feature_names) - len(transformable)
    logger(f'Filtered to {len(transformable)} transformable features (excluded {excluded_count})', "INFO")
    
    if excluded_count > 0:
        logger(f'Excluded from transformation: volatility, fwd_logret_1', "INFO")
    
    return transformable


# =================================================================================================
# Main Execution
# =================================================================================================

def main(mode='train', test_split=0):
    """Main execution function."""
    log_section(f'FEATURE TRANSFORMATION SELECTION (STAGE 07) - {mode.upper()} MODE')

    logger(f'Mode: {mode}', "INFO")
    if mode == 'test':
        logger(f'Test split: {test_split}', "INFO")
    logger('', "INFO")

    # Create Spark session (uses default 8GB driver memory and jar path)
    logger('', "INFO")
    logger('Initializing Spark session...', "INFO")
    logger('This may take 10-30 seconds on first run...', "INFO")
    spark = create_spark_session(
        app_name="FeatureTransformSelection",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI
    )
    logger('Spark session created successfully', "INFO")
    
    try:
        # Get feature names from first split using aggregation
        logger('', "INFO")
        logger('Identifying features...', "INFO")
        first_split = f"{INPUT_COLLECTION_PREFIX}0{INPUT_COLLECTION_SUFFIX}"
        logger(f'Reading feature names from collection: {first_split}', "INFO")
        
        # FIXED: Use aggregation-based function to avoid Spark schema inference issues
        all_feature_names = identify_feature_names_from_collection(
            spark=spark,
            db_name=DB_NAME,
            collection=first_split
        )
        logger(f'Total features in split collections: {len(all_feature_names)}', "INFO")
        
        # Filter to transformable features only
        feature_names = filter_transformable_features(all_feature_names)

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

        # ============================================================================
        # TEST MODE: Fit selected transforms on split_0 (ONE PASS)
        # ============================================================================
        if mode == 'test':
            logger(f'Found {len(split_ids)} split collections', "INFO")
            logger(f'TEST MODE: Fitting selected transforms on split_{test_split}', "INFO")
            logger('', "INFO")

            # Load selected transformations from train mode artifacts
            import json
            aggregation_dir = Path(REPO_ROOT) / 'artifacts' / 'feature_transformation' / 'aggregation'
            final_transforms_path = aggregation_dir / 'final_transforms.json'

            if not final_transforms_path.exists():
                raise FileNotFoundError(
                    f"Train mode transformations not found at {final_transforms_path}\n"
                    f"Please run Stage 7 in train mode first: python scripts/07_feature_transform.py --mode train"
                )

            with open(final_transforms_path, 'r') as f:
                final_transforms = json.load(f)

            logger(f'Loaded {len(final_transforms)} selected transforms from train mode', "INFO")
            logger(f'From: {final_transforms_path}', "INFO")
            logger('', "INFO")

            # Filter to only transformable features present in data
            final_transforms_filtered = {
                feat: transform for feat, transform in final_transforms.items()
                if feat in feature_names
            }

            logger(f'Will fit {len(final_transforms_filtered)} transforms on split_{test_split}', "INFO")
            logger('=' * 80, "INFO")
            logger('FITTING SELECTED TRANSFORMATIONS (ONE PASS)', "INFO")
            logger('This fits only the best transforms from train mode on 100% of split_0 data', "INFO")
            logger('=' * 80, "INFO")
            logger('', "INFO")

            # Fit selected transforms on full split_0 data (one pass)
            collection_name = f"{INPUT_COLLECTION_PREFIX}{test_split}{INPUT_COLLECTION_SUFFIX}"
            fitted_params = fit_selected_transforms_on_full_data(
                spark=spark,
                db_name=DB_NAME,
                collection=collection_name,
                feature_names=all_feature_names,
                final_transforms=final_transforms_filtered
            )

            # Save test mode artifacts
            test_mode_dir = Path(REPO_ROOT) / 'artifacts' / 'feature_transformation' / 'test_mode'
            test_mode_dir.mkdir(parents=True, exist_ok=True)

            # Save final transforms (copy from train mode for Stage 9)
            transforms_file = test_mode_dir / 'final_transforms.json'
            with open(transforms_file, 'w') as f:
                json.dump(final_transforms_filtered, f, indent=2)

            # Save fitted parameters (fitted on split_0)
            params_file = test_mode_dir / 'fitted_params.json'
            with open(params_file, 'w') as f:
                json.dump(fitted_params, f, indent=2)

            logger('', "INFO")
            logger(f'Saved test mode transforms to: {transforms_file}', "INFO")
            logger(f'Saved test mode fitted params to: {params_file}', "INFO")
            logger(f'Features fitted: {len(fitted_params)}', "INFO")

            log_section('TEST MODE COMPLETED (ONE PASS)')
            logger(f'Loaded selections from train mode, fitted on split_{test_split}', "INFO")
            logger(f'Transforms: {transforms_file}', "INFO")
            logger(f'Fitted params: {params_file}', "INFO")

            return 0  # Exit after test mode

        # ============================================================================
        # TRAIN MODE: Process all splits
        # ============================================================================
        logger(f'Found {len(split_ids)} split collections: {split_ids}', "INFO")
        logger(f'Processing {len(feature_names)} transformable features across {len(split_ids)} splits', "INFO")

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

        for split_id in split_ids:
            # Process split - pass both filtered and full feature lists
            split_results = processor.process_split(
                split_id=split_id,
                feature_names=feature_names,  # Transformable features only
                all_feature_names=all_feature_names  # Full list for array validation
            )
            all_split_results[split_id] = split_results

        # Aggregate results across splits
        logger('', "INFO")
        log_section('AGGREGATING RESULTS ACROSS SPLITS')
        aggregated = aggregate_across_splits(all_split_results)
        
        # Select final transformations
        logger('', "INFO")
        logger('Selecting final transformations...', "INFO")
        final_transforms = select_final_transforms(aggregated)

        # Save results
        results_dir = Path(REPO_ROOT) / 'artifacts' / 'feature_transformation'
        results_dir.mkdir(parents=True, exist_ok=True)

        import json
        results_file = results_dir / 'transformation_selection.json'
        with open(results_file, 'w') as f:
            json.dump({
                'final_transforms': final_transforms,
                'aggregated_metrics': {
                    feat: {
                        'selected_transform': final_transforms.get(feat, 'identity'),
                        'most_frequent_transform': agg['most_frequent_transform'],
                        'stability': float(agg['stability']),
                        'n_splits': agg['n_splits'],
                        'avg_scores': {k: float(v) for k, v in agg['avg_scores'].items()},
                        'frequency_count': agg['frequency_count']
                    }
                    for feat, agg in aggregated.items()
                }
            }, f, indent=2)
        
        logger(f'Results saved to: {results_file}', "INFO")
        
        log_section('TRANSFORMATION SELECTION COMPLETED')
        logger(f'Selected transformations for {len(final_transforms)} features', "INFO")
        logger(f'Excluded features (keep original scale): volatility, fwd_logret_1', "INFO")
        
    except Exception as e:
        logger(f'Error during transformation selection: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature transformation selection')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Pipeline mode: train (all splits) or test (split_0 only)')
    parser.add_argument('--test-split', type=int, default=0,
                       help='Split ID to use in test mode (default: 0)')
    args = parser.parse_args()

    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    main(mode=args.mode, test_split=args.test_split)