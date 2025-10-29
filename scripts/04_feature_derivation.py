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
from src.hand_crafted_features import (
    FeatureOrchestrator,
    BatchLoader,
    PriceFeatures,
    VolatilityFeatures,
    DepthFeatures,
    ForwardReturnsCalculator
)

# =================================================================================================
# Configuration
# =================================================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
INPUT_COLLECTION = "input"
OUTPUT_COLLECTION = "output"

CONFIG = {
    'forward_horizons': [2, 3, 4, 5, 10, 20, 40, 60, 120, 240],
    'historical_lags': [1, 2, 3, 4, 5, 10, 20, 40, 60, 120, 240],
    'variance_half_life': 20,
    'depth_bands': [5, 15, 50, 0],  # very-near, near, middle, whole book
    'decision_lag': 1,
    'required_past_hours': 3,
    'required_future_hours': 3,
}

ADDITIONAL_SPARK_CONFIGS = {
    "spark.network.timeout": "300s",
    "spark.executor.heartbeatInterval": "60s",
    "spark.mongodb.connection.timeout.ms": "30000",
    "spark.mongodb.socket.timeout.ms": "120000",
    "spark.mongodb.write.retryWrites": "true"
}
# Create Spark session
spark = create_spark_session(
    app_name="Stage4_FeatureEngineering",
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    driver_memory="8g",
    additional_configs=ADDITIONAL_SPARK_CONFIGS
)

# =================================================================================================
# Pipeline
# =================================================================================================

def run_feature_engineering_pipeline():
    """
    Executes feature engineering pipeline.
    """
    log_section("LOB Feature Engineering Pipeline")
    logger(f'Input Collection: {INPUT_COLLECTION}', level="INFO")
    logger(f'Output Collection: {OUTPUT_COLLECTION}', level="INFO")
    log_section("", char="-")
    logger(f'Forward Horizons: {CONFIG["forward_horizons"]}', level="INFO")
    logger(f'Historical Lags: {CONFIG["historical_lags"]}', level="INFO")
    logger(f'Depth Bands: {CONFIG["depth_bands"]}', level="INFO")
    log_section("", char="=")
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(
        spark=spark,
        db_name=DB_NAME,
        input_collection=INPUT_COLLECTION,
        output_collection=OUTPUT_COLLECTION,
        config=CONFIG
    )
    
    # Load raw LOB data
    log_section("Stage 1: Loading Raw LOB Data")
    orchestrator.load_raw_data()
    
    # Determine processable hours
    log_section("Stage 2: Determining Processable Hours")
    orchestrator.determine_processable_hours()
    
    # Process hourly batches
    log_section("Stage 3: Processing Hourly Batches")
    orchestrator.process_all_batches()
    
    log_section("Pipeline Complete")

# =================================================================================================
# Main Entry Point
# =================================================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Checks if runned from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    
    try:
        run_feature_engineering_pipeline()
        
        total_time = time.time() - start_time
        logger(f'Total execution time: {total_time:.2f} seconds', level="INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', level="ERROR")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()