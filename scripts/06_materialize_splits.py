"""
Split Materialization Script

Materializes CPCV splits from standardized LOB data into separate collections.

Usage:
    python scripts/materialize_splits.py
"""

import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from pyspark.sql import SparkSession
from src.utils.logging import logger
from src.utils.spark import create_spark_session
from src.split_materialization import SplitMaterializer


def main():
    """Main execution function."""
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    
    DB_NAME = "raw"
    INPUT_COLLECTION = "input"  # Output from Stage 5 (standardization)
    
    # Split materialization configuration
    CONFIG = {
        "max_splits": 5,  # Materialize only first N splits (None for all)
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
    
    logger('=' * 60, "INFO")
    logger('SPLIT MATERIALIZATION PIPELINE', "INFO")
    logger('=' * 60, "INFO")
    logger(f'Database: {DB_NAME}', "INFO")
    logger(f'Input Collection: {INPUT_COLLECTION}', "INFO")
    logger(f'Max Splits: {CONFIG["max_splits"]}', "INFO")
    logger(f'Create Test Collection: {CONFIG["create_test_collection"]}', "INFO")
    logger('=' * 60, "INFO")
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"],
        db_name=DB_NAME,
        mongo_uri=SPARK_CONFIG["mongo_uri"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        jar_files_path=SPARK_CONFIG["jar_files_path"]
    )
    
    try:
        # =====================================================================
        # SPLIT MATERIALIZATION
        # =====================================================================
        
        # Initialize materializer
        materializer = SplitMaterializer(
            spark=spark,
            db_name=DB_NAME,
            input_collection=INPUT_COLLECTION,
            config=CONFIG
        )
        
        # Step 1: Discover splits
        logger('Step 1: Discovering splits...', "INFO")
        materializer.discover_splits()
        
        # Step 2: Discover processable hours
        logger('Step 2: Discovering processable hours...', "INFO")
        materializer.discover_hours()
        
        # Step 3: Materialize all splits
        logger('Step 3: Materializing splits...', "INFO")
        materializer.materialize_all_splits()
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        
        logger('=' * 60, "INFO")
        logger('SPLIT MATERIALIZATION COMPLETED SUCCESSFULLY', "INFO")
        logger('=' * 60, "INFO")
        
        # Log created collections
        collections = materializer.get_split_collections()
        logger(f'Created {len(collections)} collections:', "INFO")
        for collection in collections:
            logger(f'  - {collection}', "INFO")
        
    except Exception as e:
        logger(f'Error during split materialization: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up
        spark.stop()
        logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    main()