"""
Stylized Facts Testing Pipeline (ENHANCED VERSION)

Tests stylized facts (stationarity, normality, autocorrelation, etc.) on representative
windows from each CPCV split.

ENHANCEMENTS:
- Added EnhancedResultsAggregator for statistical confidence measures
- Provides mean, variance, and 95% confidence intervals
- Outputs easy-to-analyze CSV with per-feature statistics
- Backward compatible with original outputs

Place this script in: rdl-lob/scripts/
Run with: python scripts/09_test_stylized_facts.py
"""

import os
import sys
import time
import json
from pathlib import Path
import yaml

# ============================================================================
# Path Setup
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# Unicode/MLflow Fix for Windows - MUST BE FIRST!
# =================================================================================================
# Fix Windows console encoding to handle Unicode characters (fixes MLflow emoji errors)
if sys.platform == 'win32':
    # Set environment variables for UTF-8 support
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
    os.environ['PYTHONUTF8'] = '1'
    
    # Reconfigure stdout/stderr to use UTF-8 with error replacement
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

# Patch MLflow to remove emoji that causes Windows encoding errors
try:
    from mlflow.tracking._tracking_service import client as mlflow_client
    
    _original_log_url = mlflow_client.TrackingServiceClient._log_url
    
    def _patched_log_url(self, run_id):
        try:
            run = self.get_run(run_id)
            run_name = run.info.run_name or run_id
            run_url = self._get_run_url(run.info.experiment_id, run_id)
            sys.stdout.write(f"[RUN] View run {run_name} at: {run_url}\n")
            sys.stdout.flush()
        except:
            pass
    
    mlflow_client.TrackingServiceClient._log_url = _patched_log_url
except:
    pass
# =================================================================================================


# Import utilities
from src.utils import (
    create_spark_session,
    logger,
    log_section
)

# Import stylized facts modules
from src.stylized_facts import (
    StreamingStylizedFactsPipeline,
    ResultsAggregator,
    EnhancedResultsAggregator  # NEW: For statistical confidence measures
)

# ============================================================================
# Configuration
# ============================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"  # Change to your database name
METADATA_PATH = Path(REPO_ROOT) / "artifacts" / "fold_assignment" / "reproducibility.yaml"

# Default output directory (can be overridden by command-line argument)
DEFAULT_OUTPUT_DIR = Path(REPO_ROOT) / "artifacts" / "stylized_facts"

CONFIG = {
    'forecast_horizon': 240,
    'significance_level': 0.05
}

# NEW: Enhanced statistics configuration
ENHANCED_CONFIG = {
    'confidence_level': 0.95,  # 95% confidence intervals
    'enable_enhanced_stats': True  # Set to False to disable enhanced statistics
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

def run_stylized_facts_pipeline(output_dir: Path = None):
    """
    Execute stylized facts testing pipeline.
    
    Args:
        output_dir: Output directory for results. If None, uses DEFAULT_OUTPUT_DIR.
    """
    start_time = time.time()
    
    # Use provided output_dir or default
    OUTPUT_DIR = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    
    # Create Spark session
    spark = create_spark_session(
        app_name="StylizedFactsTesting",
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        driver_memory="4g"
    )
    
    try:
        log_section("Stylized Facts Testing Pipeline (ENHANCED)")
        logger(f"Database: {DB_NAME}", "INFO")
        logger(f"Metadata: {METADATA_PATH}", "INFO")
        logger(f"Output directory: {OUTPUT_DIR}", "INFO")
        logger(f"Forecast horizon: {CONFIG['forecast_horizon']}", "INFO")
        logger(f"Significance level: {CONFIG['significance_level']}", "INFO")
        logger(f"Enhanced statistics: {'ENABLED' if ENHANCED_CONFIG['enable_enhanced_stats'] else 'DISABLED'}", "INFO")
        if ENHANCED_CONFIG['enable_enhanced_stats']:
            logger(f"Confidence level: {ENHANCED_CONFIG['confidence_level'] * 100}%", "INFO")
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
        # Step 6: Aggregate Results (Standard)
        # ====================================================================
        log_section("Step 6: Aggregating Results (Standard Summary)")
        
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
        # Step 6b: Enhanced Statistical Analysis (NEW)
        # ====================================================================
        
        if ENHANCED_CONFIG['enable_enhanced_stats']:
            logger("", "INFO")
            log_section("Step 6b: Computing Enhanced Statistics with Confidence Intervals")
            
            # Use enhanced aggregator for statistical confidence measures
            enhanced_aggregator = EnhancedResultsAggregator(
                OUTPUT_DIR, 
                confidence_level=ENHANCED_CONFIG['confidence_level']
            )
            
            # Compute enhanced statistics
            logger("Computing mean, variance, and confidence intervals...", "INFO")
            enhanced_stats = enhanced_aggregator.compute_enhanced_statistics(combined)
            
            # Create feature summary DataFrame (easy to analyze)
            logger("Creating per-feature summary DataFrame...", "INFO")
            feature_df = enhanced_aggregator.create_feature_summary_dataframe(enhanced_stats)
            
            # Compute violations with confidence intervals
            logger("Identifying violations with statistical confidence...", "INFO")
            enhanced_violations = enhanced_aggregator.identify_violations(combined)
            
            # Save enhanced outputs
            logger("Saving enhanced summary outputs...", "INFO")
            enhanced_aggregator.save_enhanced_summary(enhanced_stats, feature_df)
            
            # Also save enhanced violations
            enhanced_violations_path = OUTPUT_DIR / "enhanced_violations.json"
            with open(enhanced_violations_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_violations, f, indent=2)
            logger(f"  Saved enhanced violations: {enhanced_violations_path}", "INFO")
            
            logger("", "INFO")
            logger("Enhanced statistics computed successfully!", "INFO")
            logger(f"Confidence level used: {ENHANCED_CONFIG['confidence_level'] * 100}%", "INFO")
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
        logger(f"  Raw data:", "INFO")
        logger(f"    - summary_all_splits.csv (combined raw results)", "INFO")

        if ENHANCED_CONFIG['enable_enhanced_stats']:
            logger(f"  Enhanced summary (with statistical confidence):", "INFO")
            logger(f"    - enhanced_summary_statistics.json", "INFO")
            logger(f"    - enhanced_summary_by_feature.csv ← Use this for analysis!", "INFO")
            logger(f"    - enhanced_violations.json", "INFO")
        
        log_section("", char="=")
        
    except Exception as e:
        logger(f"ERROR: {str(e)}", "ERROR")
        import traceback
        logger(traceback.format_exc(), "DEBUG")
        raise
    
    finally:
        # Only stop Spark if not orchestrated
        if not is_orchestrated:
            spark.stop()
            logger('Spark session stopped', "INFO")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Stylized Facts Testing Pipeline (Enhanced)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (relative to repo root)')
    parser.add_argument('--no-enhanced-stats', action='store_true',
                        help='Disable enhanced statistical analysis')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for intervals (default: 0.95)')
    args = parser.parse_args()
    
    # Update configuration based on arguments
    if args.no_enhanced_stats:
        ENHANCED_CONFIG['enable_enhanced_stats'] = False
        logger("Enhanced statistics disabled via command-line argument", "INFO")
    
    if args.confidence_level:
        if 0 < args.confidence_level < 1:
            ENHANCED_CONFIG['confidence_level'] = args.confidence_level
        else:
            logger(f"Invalid confidence level: {args.confidence_level}. Using default 0.95", "WARNING")
    
    # Convert output_dir to Path if provided
    output_dir = None
    if args.output_dir:
        output_dir = Path(REPO_ROOT) / args.output_dir
        logger(f"Using output directory from argument: {output_dir}", "INFO")
    else:
        logger(f"Using default output directory: {DEFAULT_OUTPUT_DIR}", "INFO")
    
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
    
    try:
        run_stylized_facts_pipeline(output_dir=output_dir)
    except Exception as e:
        logger(f"Pipeline failed: {str(e)}", "ERROR")
        raise