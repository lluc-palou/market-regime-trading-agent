"""
Split Materialization Script (Stage 06)

Materializes CPCV splits from standardized LOB data into separate collections.
Projects features needed for downstream stages.

Projection Strategy:
- KEEP in features array: microprice, volatility, depth (10), historical (6), fwd_logret_1
  Total: 18 features
- EXCLUDE from features array: mid_price, log_return, variance_proxy, spread
  (These are intermediate features only needed for calculation)

Next stages will handle volatility and fwd_logret_1 specially:
- They remain in features array but excluded from transformation/standardization

Usage:
    python scripts/06_materialize_splits.py
"""

import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# CRITICAL: Windows UTF-8 Fix - MUST BE BEFORE OTHER IMPORTS!
# =================================================================================================
if sys.platform == 'win32':
    # Force UTF-8 encoding for all I/O operations
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Reconfigure stdout/stderr to use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
# =================================================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array, lit

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session
from src.split_materialization import SplitMaterializer


# =================================================================================================
# Feature Projection Configuration
# =================================================================================================

# Features to EXCLUDE from projection (intermediate calculations not needed downstream)
EXCLUDE_FROM_PROJECTION = [
    'mid_price',        # Used to calculate log_return, then not needed
    'log_return',       # Used to calculate hist_logret_*, then not needed
    'variance_proxy',   # Used to calculate volatility, then not needed
    'spread',           # Nearly constant, not informative
]

# NOTE: volatility and fwd_logret_1 ARE kept in features array
# They will be excluded from transformation/standardization in Stages 07-11


def filter_projected_features(feature_names):
    """
    Filter features to project to split collections.
    
    Keeps: microprice, volatility, depth (10), historical (6), fwd_logret_1 = 18 features
    Excludes: mid_price, log_return, variance_proxy, spread = 4 features
    
    Args:
        feature_names: List of all feature names (should be 22 total)
        
    Returns:
        List of features to project (should be 18 features)
    """
    projected = [
        feat for feat in feature_names 
        if feat not in EXCLUDE_FROM_PROJECTION
    ]
    
    excluded_count = len(feature_names) - len(projected)
    
    logger(f'Feature projection: {len(projected)} features (excluded {excluded_count})', "INFO")
    if excluded_count > 0:
        excluded_list = [f for f in feature_names if f in EXCLUDE_FROM_PROJECTION]
        logger(f'Excluded intermediate features: {", ".join(excluded_list)}', "INFO")
    
    return projected


def apply_feature_projection(df, projected_features):
    """
    Apply feature projection to DataFrame by filtering feature arrays.
    Uses native PySpark array indexing (much faster than UDFs).

    ENHANCED: Extracts fwd_logret_1 (forward log-return) as independent field 'target'
    for easier access in downstream stages.

    Args:
        df: PySpark DataFrame with feature_names and features columns
        projected_features: List of features to keep (18 features)

    Returns:
        DataFrame with filtered features and separate 'target' field
    """
    logger('Applying feature projection to DataFrame...', "INFO")

    # Get feature names from first row to determine indices
    first_row = df.select('feature_names').first()
    all_feature_names = first_row['feature_names']

    # Find index of fwd_logret_1 for extraction as independent field
    fwd_logret_idx = None
    for feat in ['fwd_logret_1', 'fwd_logret_01']:  # Try both naming conventions
        try:
            fwd_logret_idx = all_feature_names.index(feat)
            logger(f'Found forward log-return at index {fwd_logret_idx}: {feat}', "INFO")
            break
        except ValueError:
            continue

    if fwd_logret_idx is None:
        logger('WARNING: fwd_logret_1 not found in features, target field will not be created', "WARNING")

    # Extract fwd_logret_1 as independent 'target' field BEFORE projection
    if fwd_logret_idx is not None:
        df = df.withColumn('target', col('features')[fwd_logret_idx])
        logger('Extracted fwd_logret_1 as independent field "target"', "INFO")

    # Remove fwd_logret_1 from projected_features list (will be separate field)
    projected_features_without_target = [f for f in projected_features
                                         if not f.startswith('fwd_logret')]

    logger(f'Features after target extraction: {len(projected_features_without_target)} (removed forward returns)', "INFO")

    # Build index mapping: projected feature -> array index
    feature_indices = []
    for feat in projected_features_without_target:
        try:
            idx = all_feature_names.index(feat)
            feature_indices.append(idx)
        except ValueError:
            logger(f'Warning: Feature {feat} not found in data, using 0.0', "WARNING")
            feature_indices.append(None)  # Will handle missing features

    # Create projected array using native PySpark array indexing
    # Build array by indexing into the features array
    projected_values = []
    for idx in feature_indices:
        if idx is not None:
            projected_values.append(col('features')[idx])
        else:
            projected_values.append(lit(0.0))

    # Replace features column with projected array (without forward returns)
    df = df.withColumn('features', array(*projected_values))

    # Update feature_names to projected list (without forward returns)
    projected_array = array([lit(name) for name in projected_features_without_target])
    df = df.withColumn('feature_names', projected_array)

    logger(f'Projection applied: {len(projected_features_without_target)} features in array', "INFO")
    if fwd_logret_idx is not None:
        logger('Target field: fwd_logret_1 extracted as separate "target" column', "INFO")

    return df


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    
    DB_NAME = "raw"
    INPUT_COLLECTION = "input"  # Output from Stage 5 (LOB standardization)
    
    # Split materialization configuration
    CONFIG = {
        "max_splits": None,  # Materialize only first split
        "create_test_collection": True,  # Create separate test_data collection
    }
    
    # Spark configuration
    SPARK_CONFIG = {
        "app_name": "SplitMaterialization",
        "mongo_uri": "mongodb://127.0.0.1:27017/",
        "driver_memory": "8g",
        "jar_files_path": "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/",
    }
    
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    
    log_section('SPLIT MATERIALIZATION WITH FEATURE PROJECTION (STAGE 06)')
    
    logger(f'Database: {DB_NAME}', "INFO")
    logger(f'Input Collection: {INPUT_COLLECTION}', "INFO")
    logger(f'Max Splits: {CONFIG["max_splits"]} (materialize first split only)', "INFO")
    logger(f'Create Test Collection: {CONFIG["create_test_collection"]}', "INFO")
    
    # Create Spark session
    logger('', "INFO")
    logger('Initializing Spark...', "INFO")
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"],
        db_name=DB_NAME,
        mongo_uri=SPARK_CONFIG["mongo_uri"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        jar_files_path=SPARK_CONFIG["jar_files_path"]
    )
    
    try:
        # =====================================================================
        # FEATURE PROJECTION SETUP
        # =====================================================================
        
        logger('', "INFO")
        log_section('DETERMINING FEATURE PROJECTION')
        
        # Load sample to get feature names
        sample_df = (
            spark.read.format("mongodb")
            .option("database", DB_NAME)
            .option("collection", INPUT_COLLECTION)
            .load()
            .limit(1)
        )
        
        if sample_df.count() == 0:
            raise ValueError(f'Input collection {INPUT_COLLECTION} is empty!')
        
        # Get feature names
        first_row = sample_df.first()
        all_feature_names = first_row['feature_names']
        
        logger(f'Total features in input: {len(all_feature_names)}', "INFO")
        
        # Determine projected features
        projected_features = filter_projected_features(all_feature_names)
        
        logger('', "INFO")
        logger(f'Projected features to split collections ({len(projected_features)} total before extraction):', "INFO")
        logger('  Features array (17 features after fwd_logret_1 extraction):', "INFO")
        logger('    - microprice (1)', "INFO")
        logger('    - volatility (1) - kept in array, excluded from transform/std', "INFO")
        logger('    - depth features (10)', "INFO")
        logger('    - historical returns (6)', "INFO")
        logger('  Extracted as separate "target" field:', "INFO")
        logger('    - fwd_logret_1 (1) - independent field for easier access', "INFO")
        
        # =====================================================================
        # SPLIT MATERIALIZATION
        # =====================================================================
        
        logger('', "INFO")
        log_section('MATERIALIZING SPLITS')

        # CRITICAL: Create timestamp index on input collection for efficient hourly queries
        # Without this index, each hourly query performs a full collection scan O(N)
        # With index: O(log N + matches) - reduces processing time dramatically
        logger('Creating timestamp index on input collection...', "INFO")
        from pymongo import MongoClient, ASCENDING
        mongo_uri = spark.sparkContext.getConf().get('spark.mongodb.read.connection.uri', 'mongodb://127.0.0.1:27017/')
        client = MongoClient(mongo_uri)
        db = client[DB_NAME]
        input_coll = db[INPUT_COLLECTION]

        # Check if index already exists
        existing_indexes = list(input_coll.list_indexes())
        has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

        if not has_timestamp_index:
            logger('Creating index on timestamp field...', "INFO")
            input_coll.create_index([("timestamp", ASCENDING)], background=False)
            logger('Timestamp index created successfully', "INFO")
        else:
            logger('Timestamp index already exists', "INFO")

        client.close()
        logger('', "INFO")

        # Initialize materializer
        # NOTE: This assumes SplitMaterializer accepts these parameters
        # You may need to modify the materializer class to accept projection parameters
        materializer = SplitMaterializer(
            spark=spark,
            db_name=DB_NAME,
            input_collection=INPUT_COLLECTION,
            config=CONFIG
        )
        
        # Apply projection during materialization
        # This will be done in the batch processor
        materializer.projected_features = projected_features
        materializer.apply_projection_func = apply_feature_projection
        
        # Step 1: Discover splits
        logger('Step 1: Discovering splits...', "INFO")
        materializer.discover_splits()
        
        # Step 2: Discover processable hours
        logger('Step 2: Discovering processable hours...', "INFO")
        materializer.discover_hours()
        
        # Step 3: Materialize all splits (with projection applied)
        logger('Step 3: Materializing splits with feature projection...', "INFO")
        materializer.materialize_all_splits()
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        
        logger('', "INFO")
        log_section('SPLIT MATERIALIZATION COMPLETED')
        
        # Log created collections
        collections = materializer.get_split_collections()
        logger(f'Created {len(collections)} collections with {len(projected_features)} features each:', "INFO")
        for collection in collections:
            logger(f'  - {collection}', "INFO")
        
        logger('', "INFO")
        logger('Feature projection summary:', "INFO")
        logger(f'  Input features: {len(all_feature_names)}', "INFO")
        logger(f'  Features array: {len(projected_features) - 1} features (17 total)', "INFO")
        logger(f'  Target field: fwd_logret_1 extracted as independent "target" column', "INFO")
        logger(f'  Excluded from features: {", ".join(EXCLUDE_FROM_PROJECTION)}', "INFO")
        logger('', "INFO")
        logger('Document structure in split collections:', "INFO")
        logger('  - features: [microprice, volatility, depth (10), historical (6)] = 17 features', "INFO")
        logger('  - target: fwd_logret_1 (separate field for easier access)', "INFO")
        logger('  - feature_names: array matching features (17 names)', "INFO")
        logger('', "INFO")
        logger('NOTE: Next stages (07-08, 10-11) will transform/standardize:', "INFO")
        logger('  - microprice, depth features (10), historical returns (6) = 16 features', "INFO")
        logger('  - volatility excluded (keep original scale)', "INFO")
        logger('  - target field already separate (not in features array)', "INFO")
        
    except Exception as e:
        logger(f'Error during split materialization: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    main()