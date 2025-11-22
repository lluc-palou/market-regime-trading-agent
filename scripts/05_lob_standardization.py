import os
import sys
import time

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Import utilities
from src.utils.spark import create_spark_session
from src.utils.logging import logger, log_section

# Import standardization classes
from src.lob_standardization import StandardizationOrchestrator

# =================================================================================================
# Configuration
# =================================================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
INPUT_COLLECTION = "input"  # Input from feature engineering stage
OUTPUT_COLLECTION = "output"

CONFIG = {
    # Quantization parameters
    'B': 1000,                      # Number of bins (output will be B+1 bins)
    'delta': 1000.0,               # Price clipping threshold (std deviations)
    'epsilon': 1.0,                # Minimum price spacing near zero
    
    # Standardization parameters
    'eps': 1e-8,                   # Small constant to avoid division by zero
    'min_denom': 1e-6,             # Minimum denominator value
    
    # Batch processing
    'required_past_hours': 1,      # Hours of past context for volatility
    
    # Pipeline mode
    'volume_coverage_analysis': False,  # Set to True for coverage analysis
}

# Additional Spark configurations
ADDITIONAL_SPARK_CONFIGS = {
    "spark.network.timeout": "300s",
    "spark.executor.heartbeatInterval": "60s",
    "spark.mongodb.connection.timeout.ms": "30000",
    "spark.mongodb.socket.timeout.ms": "120000",
    "spark.mongodb.write.retryWrites": "true",
}

# Create Spark session (uses default 8GB driver memory)
spark = create_spark_session(
    app_name="Stage5_LOB_Standardization",
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    additional_configs=ADDITIONAL_SPARK_CONFIGS
)

# =================================================================================================
# Pipeline
# =================================================================================================

def run_standardization_pipeline():
    """
    Executes LOB standardization pipeline.
    """
    log_section("LOB Standardization Pipeline")
    logger(f'Input Collection: {INPUT_COLLECTION}', "INFO")
    logger(f'Output Collection: {OUTPUT_COLLECTION}', "INFO")
    log_section("", char="-")
    logger(f'Number of bins (B): {CONFIG["B"]} (output: {CONFIG["B"]+1} bins)', "INFO")
    logger(f'Price clipping threshold (delta): {CONFIG["delta"]} std', "INFO")
    logger(f'Minimum price spacing (epsilon): {CONFIG["epsilon"]}', "INFO")
    logger(f'Volume coverage analysis mode: {CONFIG["volume_coverage_analysis"]}', "INFO")
    log_section("", char="=")

    # CRITICAL: Create timestamp index on input collection for efficient hourly queries
    # Without this index, each hourly query performs a full collection scan O(N)
    # With index: O(log N + matches) - reduces processing time dramatically
    logger('Creating timestamp index on input collection...', "INFO")
    from pymongo import MongoClient, ASCENDING
    client = MongoClient(MONGO_URI)
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
    log_section("", char="-")

    # Initialize orchestrator
    orchestrator = StandardizationOrchestrator(
        spark=spark,
        db_name=DB_NAME,
        input_collection=INPUT_COLLECTION,
        output_collection=OUTPUT_COLLECTION,
        config=CONFIG
    )
    
    # Load raw LOB data
    log_section("Stage 1: Loading Split LOB Data")
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
        run_standardization_pipeline()
        
        total_time = time.time() - start_time
        logger(f'Total execution time: {total_time:.2f} seconds', "INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()