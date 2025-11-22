import os
import sys
import time
from pathlib import Path
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

# Import validation classes
from src.validation import (
    TimelineAnalyzer,
    FoldsDivider,
    CPCVsplitGenerator,
    MetadataHandler,
    DataStamper
)

# Import representative windows extractor
from src.stylized_facts import RepresentativeWindowsExtractor

# =================================================================================================
# Configuration
# =================================================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
INPUT_COLLECTION = "input"
OUTPUT_COLLECTION = "output"

CONFIG = {
    'experiment_id': None,
    'master_seed': 42,
    
    'temporal_params': {
        'sampling_interval_seconds': 30,
        'context_length_samples': 120,
        'forecast_horizon_steps': 120,
        'purge_length_samples': 120,
        'embargo_length_samples': 120,
    },
    
    'train_test_split': {
        'test_ratio': 0.20,
    },
    
    'cpcv': {
        'n_folds': 8,
        'k_validation_folds': 2,
    },
    
    'stylized_facts': {
        'window_length_samples': 2500,
        'edge_margin_samples': 120,
    }
}

# Create Spark session
spark = create_spark_session(
    app_name="CPCV_Split",
    mongo_uri=MONGO_URI,
    db_name=DB_NAME
)

# =================================================================================================
# Pipeline
# =================================================================================================

def run_cpcv_pipeline():
    """
    Executes CPCV pipeline stages.
    """
    # Generate experiment ID
    experiment_id = f"cpcv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    CONFIG['experiment_id'] = experiment_id
    METADATA_PATH = Path(REPO_ROOT) / "artifacts" / "fold_assignment"
    METADATA_PATH.mkdir(parents=True, exist_ok=True)
    metadata_file = os.path.join(METADATA_PATH, 'reproducibility.yaml')
    
    log_section("Dataset Split and CPCV Method")
    logger(f'Experiment ID: {experiment_id}', level="INFO")
    logger(f'Input Collection: {INPUT_COLLECTION}', level="INFO")
    logger(f'Output Collection: {OUTPUT_COLLECTION}', level="INFO")
    logger(f'Metadata File: {metadata_file}', level="INFO")
    log_section("", char="-")
    logger(f'Test Ratio: {CONFIG["train_test_split"]["test_ratio"]*100}%', level="INFO")
    logger(f'CPCV: {CONFIG["cpcv"]["n_folds"]} folds, k={CONFIG["cpcv"]["k_validation_folds"]}', level="INFO")
    logger(f'Purge/Embargo length: {CONFIG["temporal_params"]["purge_length_samples"]} samples', level="INFO")
    log_section("", char="=")
    
    # Stage 1: Timeline Analysis
    log_section("Stage 1: Timeline Analysis and Main Split")
    
    analyzer = TimelineAnalyzer(CONFIG, spark, DB_NAME)
    analyzer.load_timestamps(INPUT_COLLECTION)
    analyzer.calculate_usable_range()
    train_warmup_ts, train_ts, embargo_ts, test_ts, test_horizon_ts = analyzer.split_train_test()
    
    # Stage 2: Folds and Splits Construction
    log_section("Stage 2: Folds and Splits Construction")
    
    folds_divider = FoldsDivider(train_warmup_ts, train_ts, embargo_ts, test_ts, test_horizon_ts, CONFIG)
    folds = folds_divider.divide_timeline()
    
    CPCV_generator = CPCVsplitGenerator(folds, train_ts, CONFIG)
    CPCV_splits = CPCV_generator.generate_splits()
    
    # MODIFIED: Extract windows AFTER split generation, pass splits to extractor
    log_section("Stage 2b: Representative Windows Extraction (Per Split, Train Folds Only)")
    windows_extractor = RepresentativeWindowsExtractor(folds, CPCV_splits, CONFIG)
    representative_windows = windows_extractor.extract_windows_per_split()
    
    # Stage 3: Metadata Handling
    log_section("Stage 3: Metadata Handling")
    
    metadata_handler = MetadataHandler(CONFIG)
    metadata = metadata_handler.create_metadata(analyzer, folds, CPCV_splits, representative_windows)
    metadata_handler.write_metadata(metadata, metadata_file)
    
    # Stage 4: Dataset Samples Role Stamping
    log_section("Stage 4: Dataset Samples Role Stamping")

    # CRITICAL: Create index on timestamp field for efficient hourly queries
    # Without this index, each hourly query performs a full collection scan O(N)
    # With index: O(log N + matches) - reduces 77-file case from hours to minutes
    logger('Creating timestamp index on input collection...', level="INFO")
    from pymongo import MongoClient, ASCENDING
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    input_coll = db[INPUT_COLLECTION]

    # Check if index already exists
    existing_indexes = list(input_coll.list_indexes())
    has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

    if not has_timestamp_index:
        logger('Creating index on timestamp field...', level="INFO")
        input_coll.create_index([("timestamp", ASCENDING)], background=False)
        logger('Timestamp index created successfully', level="INFO")
    else:
        logger('Timestamp index already exists', level="INFO")

    client.close()

    data_stamper = DataStamper(metadata, folds, spark, DB_NAME)
    data_stamper.process_batches(INPUT_COLLECTION, OUTPUT_COLLECTION)
    
    log_section("Pipeline Complete")

# =================================================================================================
# Main Entry Point
# =================================================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Checks if runned from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    
    try:
        run_cpcv_pipeline()
        
        total_time = time.time() - start_time
        logger(f'Total execution time: {total_time:.2f} seconds', level="INFO")
        
    except Exception as e:
        logger(f'ERROR: {str(e)}', level="ERROR")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Only stop Spark if running standalone
        if not is_orchestrated:
            spark.stop()