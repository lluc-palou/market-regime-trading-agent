import os
import sys
import time
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Import utilities
from src.utils import (
    create_spark_session,
    logger,
    log_section
)

# Import feature engineering classes
from src.hand_crafted_features import FeatureOrchestrator

# =================================================================================================
# Configuration
# =================================================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
INPUT_COLLECTION = "input"
OUTPUT_COLLECTION = "output"

CONFIG = {
    'forward_horizons': [1],  # Immediate 1-step ahead only
    'historical_lags': [1, 2, 3, 5, 10, 20],  # Cut at 20 (white noise after)
    'variance_half_life': 20,
    'depth_bands': [5, 50],  # top_5 and top_50 only
    'decision_lag': 0,
    'required_past_hours': 3,
    'required_future_hours': 1,  # Only need 1 step ahead now
}

ADDITIONAL_SPARK_CONFIGS = {
    "spark.network.timeout": "300s",
    "spark.executor.heartbeatInterval": "60s",
    "spark.mongodb.connection.timeout.ms": "30000",
    "spark.mongodb.socket.timeout.ms": "120000",
    "spark.mongodb.write.retryWrites": "true"
}

# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    
    start_time = time.time()
    
    log_section('FEATURE DERIVATION (STAGE 04)')
    
    logger(f'Database: {DB_NAME}', "INFO")
    logger(f'Input Collection: {INPUT_COLLECTION}', "INFO")
    logger(f'Output Collection: {OUTPUT_COLLECTION}', "INFO")
    logger('', "INFO")
    logger('Configuration:', "INFO")
    logger(f'  Forward horizons: {CONFIG["forward_horizons"]}', "INFO")
    logger(f'  Historical lags: {CONFIG["historical_lags"]}', "INFO")
    logger(f'  Depth bands: {CONFIG["depth_bands"]}', "INFO")
    logger(f'  Variance half-life: {CONFIG["variance_half_life"]}', "INFO")
    
    # Create Spark session
    logger('', "INFO")
    logger('Initializing Spark session...', "INFO")
    spark = create_spark_session(
        app_name="FeatureDerivation",
        db_name=DB_NAME,
        mongo_uri=MONGO_URI,
        driver_memory="8g",
        jar_files_path="file:///C:/Users/llucp/spark_jars/",
        additional_configs=ADDITIONAL_SPARK_CONFIGS
    )
    
    try:
        # Initialize orchestrator
        logger('', "INFO")
        logger('Initializing feature orchestrator...', "INFO")
        orchestrator = FeatureOrchestrator(
            spark=spark,
            db_name=DB_NAME,
            input_collection=INPUT_COLLECTION,
            output_collection=OUTPUT_COLLECTION,
            config=CONFIG
        )
        
        # Execute feature engineering pipeline
        log_section('EXECUTING FEATURE ENGINEERING')
        
        # Step 1: Load raw data
        logger('Step 1: Loading raw LOB data...', "INFO")
        orchestrator.load_raw_data()
        
        # Step 2: Determine processable hours
        logger('', "INFO")
        logger('Step 2: Determining processable hours...', "INFO")
        orchestrator.determine_processable_hours()
        logger(f'Found {len(orchestrator.processable_hours)} processable hours', "INFO")
        
        # Step 3: Process all batches
        logger('', "INFO")
        logger('Step 3: Processing all batches...', "INFO")
        orchestrator.process_all_batches()
        
        # Log completion
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        logger('', "INFO")
        log_section('FEATURE DERIVATION COMPLETED')
        logger(f'Total time: {hours}h {minutes}m {seconds}s', "INFO")
        logger(f'Output collection: {OUTPUT_COLLECTION}', "INFO")
        
    except Exception as e:
        logger(f'Error during feature derivation: {str(e)}', "ERROR")
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