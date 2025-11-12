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
from pyspark.sql.functions import col, udf, array, lit
from pyspark.sql.types import ArrayType, DoubleType

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
    
    Args:
        df: PySpark DataFrame with feature_names and features columns
        projected_features: List of features to keep (18 features)
        
    Returns:
        DataFrame with filtered features
    """
    logger('Applying feature projection to DataFrame...', "INFO")
    
    # Create UDF to filter features based on projected list
    def filter_features_udf(names, values):
        if names is None or values is None:
            return None
        
        # Create mapping
        feature_map = dict(zip(names, values))
        
        # Project only selected features in order
        return [feature_map.get(name, 0.0) for name in projected_features]
    
    filter_udf_func = udf(filter_features_udf, ArrayType(DoubleType()))
    
    # Apply filtering
    df = df.withColumn('features', filter_udf_func(col('feature_names'), col('features')))
    
    # Update feature_names to projected list
    projected_array = array([lit(name) for name in projected_features])
    df = df.withColumn('feature_names', projected_array)
    
    logger(f'Projection applied: {len(projected_features)} features in final arrays', "INFO")
    
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
        "max_splits": 3,  # Materialize only first N splits (None for all)
        "create_test_collection": True,  # Create separate test_data collection
    }
    
    # Spark configuration
    SPARK_CONFIG = {
        "app_name": "SplitMaterialization",
        "mongo_uri": "mongodb://127.0.0.1:27017/",
        "driver_memory": "8g",
        "jar_files_path": "file:///C:/Users/llucp/spark_jars/",
    }
    
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    
    log_section('SPLIT MATERIALIZATION WITH FEATURE PROJECTION (STAGE 06)')
    
    logger(f'Database: {DB_NAME}', "INFO")
    logger(f'Input Collection: {INPUT_COLLECTION}', "INFO")
    logger(f'Max Splits: {CONFIG["max_splits"]}', "INFO")
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
        logger(f'Projected features to split collections ({len(projected_features)} total):', "INFO")
        logger('  - microprice (1)', "INFO")
        logger('  - volatility (1) - kept in array, excluded from transform/std', "INFO")
        logger('  - depth features (10)', "INFO")
        logger('  - historical returns (6)', "INFO")
        logger('  - fwd_logret_1 (1) - kept in array, excluded from transform/std', "INFO")
        
        # =====================================================================
        # SPLIT MATERIALIZATION
        # =====================================================================
        
        logger('', "INFO")
        log_section('MATERIALIZING SPLITS')
        
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
        logger(f'  Output features: {len(projected_features)}', "INFO")
        logger(f'  Excluded: {", ".join(EXCLUDE_FROM_PROJECTION)}', "INFO")
        logger('', "INFO")
        logger('NOTE: Next stages (07-08, 10-11) will further exclude:', "INFO")
        logger('  - volatility (keep original scale)', "INFO")
        logger('  - fwd_logret_1 (target, keep original scale)', "INFO")
        logger(f'  Resulting in 16 features transformed/standardized', "INFO")
        
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