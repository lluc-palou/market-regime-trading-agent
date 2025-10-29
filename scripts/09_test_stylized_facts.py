"""
Stylized Facts Testing Pipeline

Tests stylized facts (stationarity, normality, autocorrelation, etc.) on representative
windows from each CPCV split.

Place this script in: rdl-lob/scripts/
Run with: python scripts/05_stylized_facts_testing.py
"""

import os
import sys
import time
from pathlib import Path
import yaml

# ============================================================================
# Path Setup
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Import utilities
from src.utils import (
    create_spark_session,
    logger,
    log_section
)

# Import stylized facts modules
from src.stylized_facts import (
    StreamingStylizedFactsPipeline,
    ResultsAggregator
)

# ============================================================================
# Configuration
# ============================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"  # Change to your database name
METADATA_PATH = Path(REPO_ROOT) / "artifacts" / "fold_assignment" / "reproducibility.yaml"
OUTPUT_DIR = Path(REPO_ROOT) / "artifacts" / "stylized_facts_results"

CONFIG = {
    'forecast_horizon': 240,
    'significance_level': 0.05
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_metadata(metadata_path: Path) -> dict:
    """
    Load metadata from YAML file.
    
    Args:
        metadata_path: Path to reproducibility.yaml
        
    Returns:
        Metadata dictionary
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    return metadata


def get_splits_from_windows(metadata: dict) -> list:
    """
    Extract split IDs that have windows.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Sorted list of split IDs
    """
    windows = metadata.get('stylized_facts_windows', {}).get('windows', [])
    
    if not windows:
        raise ValueError("No windows found in metadata!")
    
    # Extract unique split IDs from windows
    split_ids = sorted(set(w['split_id'] for w in windows if 'split_id' in w))
    
    return split_ids


def validate_windows(metadata: dict):
    """
    Validate windows in metadata.
    
    Args:
        metadata: Metadata dictionary
    """
    log_section("Validating Windows")
    
    windows = metadata.get('stylized_facts_windows', {}).get('windows', [])
    logger(f"Total windows: {len(windows)}", "INFO")
    
    if not windows:
        raise ValueError("No windows found in metadata!")
    
    # Get window configuration
    window_length = metadata['stylized_facts_windows']['window_length_samples']
    edge_margin = metadata['stylized_facts_windows']['edge_margin_samples']
    
    logger(f"Window length: {window_length} samples", "INFO")
    logger(f"Edge margin: {edge_margin} samples", "INFO")
    logger(f"Minimum fold size: {2 * edge_margin + window_length} samples", "INFO")
    
    # Count windows by fold type
    fold_types = {}
    for w in windows:
        fold_type = w.get('fold_type', 'unknown')
        fold_types[fold_type] = fold_types.get(fold_type, 0) + 1
    
    logger(f"Windows by fold type:", "INFO")
    for fold_type, count in fold_types.items():
        logger(f"  {fold_type}: {count}", "INFO")
    
    # Get unique splits and folds
    split_ids = set(w['split_id'] for w in windows if 'split_id' in w)
    fold_ids = set(w['fold_id'] for w in windows if 'fold_id' in w)
    
    logger(f"Unique splits: {len(split_ids)}", "INFO")
    logger(f"Unique folds: {len(fold_ids)}", "INFO")
    
    # Calculate windows per split
    from collections import Counter
    split_counts = Counter(w['split_id'] for w in windows)
    windows_per_split = list(split_counts.values())
    
    logger(f"Windows per split: min={min(windows_per_split)}, "
          f"max={max(windows_per_split)}, "
          f"avg={sum(windows_per_split)/len(windows_per_split):.1f}", "INFO")
    
    log_section("", char="-")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_stylized_facts_pipeline():
    """
    Execute stylized facts testing pipeline.
    """
    start_time = time.time()
    
    # Create Spark session
    spark = create_spark_session(
        app_name="StylizedFactsTesting",
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        driver_memory="4g"
    )
    
    try:
        log_section("Stylized Facts Testing Pipeline")
        logger(f"Database: {DB_NAME}", "INFO")
        logger(f"Metadata: {METADATA_PATH}", "INFO")
        logger(f"Output directory: {OUTPUT_DIR}", "INFO")
        logger(f"Forecast horizon: {CONFIG['forecast_horizon']}", "INFO")
        logger(f"Significance level: {CONFIG['significance_level']}", "INFO")
        log_section("", char="=")
        
        # ====================================================================
        # Step 1: Load Metadata
        # ====================================================================
        log_section("Step 1: Loading Metadata")
        
        metadata = load_metadata(METADATA_PATH)
        logger(f"Loaded metadata successfully", "INFO")
        
        # ====================================================================
        # Step 2: Validate Windows
        # ====================================================================
        validate_windows(metadata)
        
        # ====================================================================
        # Step 3: Identify Splits to Test
        # ====================================================================
        log_section("Step 3: Identifying Splits to Test")
        
        # Extract split IDs from windows
        split_ids = get_splits_from_windows(metadata)
        
        logger(f"Found {len(split_ids)} splits with windows", "INFO")
        logger(f"Split IDs: {split_ids[:10]}{'...' if len(split_ids) > 10 else ''}", "INFO")
        logger(f"Split range: {min(split_ids)} to {max(split_ids)}", "INFO")
        
        if not split_ids:
            logger("ERROR: No splits to test!", "ERROR")
            return
        
        log_section("", char="-")
        
        # ====================================================================
        # Step 4: Initialize Pipeline
        # ====================================================================
        log_section("Step 4: Initializing Testing Pipeline")
        
        pipeline = StreamingStylizedFactsPipeline(
            spark=spark,
            db_name=DB_NAME,
            forecast_horizon=CONFIG['forecast_horizon'],
            output_dir=OUTPUT_DIR,
            significance_level=CONFIG['significance_level']
        )
        
        logger("Pipeline initialized successfully", "INFO")
        log_section("", char="-")
        
        # ====================================================================
        # Step 5: Run Tests on All Splits
        # ====================================================================
        log_section("Step 5: Testing All Splits")
        logger(f"Testing {len(split_ids)} splits...", "INFO")
        logger("", "INFO")
        
        results_paths = pipeline.test_all_splits(
            metadata=metadata,
            split_ids=split_ids
        )
        
        logger("", "INFO")
        logger(f"Completed testing {len(results_paths)} splits", "INFO")
        logger("", "INFO")
        
        # ====================================================================
        # Step 6: Aggregate Results
        # ====================================================================
        log_section("Step 6: Aggregating Results")
        
        if not results_paths:
            logger("WARNING: No result files to aggregate", "WARNING")
            return
        
        aggregator = ResultsAggregator(OUTPUT_DIR)
        
        # Combine all split results
        logger("Combining results from all splits...", "INFO")
        combined = aggregator.aggregate_from_files(results_paths)
        
        if combined.empty:
            logger("WARNING: No results to aggregate", "WARNING")
            return
        
        # Compute summary statistics
        logger("Computing summary statistics...", "INFO")
        summary_stats = aggregator.compute_summary_statistics(combined)
        
        # Identify violations
        logger("Identifying violations...", "INFO")
        violations = aggregator.identify_violations(combined)
        
        # Save all outputs
        logger("Saving summary outputs...", "INFO")
        aggregator.save_summary(combined, summary_stats, violations)
        
        logger("", "INFO")
        log_section("", char="-")
        
        # ====================================================================
        # Summary
        # ====================================================================
        elapsed_time = time.time() - start_time
        
        logger("", "INFO")
        log_section("Pipeline Complete")
        logger(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)", "INFO")
        logger(f"Splits tested: {len(results_paths)}", "INFO")
        logger(f"Total test results: {len(combined):,}", "INFO")
        logger(f"Features analyzed: {combined['feature_name'].nunique()}", "INFO")
        logger(f"Results saved to: {OUTPUT_DIR}", "INFO")
        logger("", "INFO")
        logger("Output files:", "INFO")
        logger(f"  - summary_all_splits.csv", "INFO")
        logger(f"  - summary_statistics.json", "INFO")
        logger(f"  - violations.json", "INFO")
        logger(f"  - summary_report.txt", "INFO")
        log_section("", char="=")
        
    except Exception as e:
        logger(f"ERROR: {str(e)}", "ERROR")
        import traceback
        logger(traceback.format_exc(), "DEBUG")
        raise
    
    finally:
        spark.stop()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    
    try:
        run_stylized_facts_pipeline()
    except Exception as e:
        logger(f"Pipeline failed: {str(e)}", "ERROR")
        raise