import os
import sys
from datetime import datetime

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
        """Patched MLflow URL logger - replaces emoji with [RUN]"""
        try:
            run = self.get_run(run_id)
            run_name = run.info.run_name or run_id
            run_url = self._get_run_url(run.info.experiment_id, run_id)
            sys.stdout.write(f"[RUN] View run {run_name} at: {run_url}\n")
            sys.stdout.flush()
        except:
            pass  # Silently skip if anything fails
    
    mlflow_client.TrackingServiceClient._log_url = _patched_log_url
except:
    pass  # MLflow not imported yet or patch failed

# =================================================================================================
# Setup paths
# =================================================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger, log_section
from src.pipeline import CyclicPipelineManager, StageRunner

# =================================================================================================
# Configuration - Edit these settings
# =================================================================================================

CONFIG = {
    # MongoDB settings
    'mongo_uri': "mongodb://127.0.0.1:27017/",
    'db_name': "raw",
    
    # Pipeline control
    'reset': False,        # Reset pipeline to archive state before running
    'start_from': 8,       # Start from stage 3-13
    'stop_at': 8,         # Stop at stage 3-13
    'skip_init': True,     # Skip initialization check (only if you know pipeline is ready)
    
    # Stylized facts testing
    'enable_stylized_facts': True,  # Enable stylized facts analysis
}

# =================================================================================================
# Stage Definitions
# =================================================================================================

STAGES = {
    3: {
        "name": "Data Splitting",
        "script": "03_data_splitting.py",
        "swap_after": True,
        "description": "Split data into temporal folds using CPCV",
        "stage_type": "pipeline"
    },
    4: {
        "name": "Feature Engineering",
        "script": "04_feature_derivation.py",
        "swap_after": True,
        "description": "Derive features from LOB snapshots",
        "stage_type": "pipeline"
    },
    5: {
        "name": "LOB Standardization",
        "script": "05_lob_standardization.py",
        "swap_after": True,
        "description": "Standardize LOB features",
        "stage_type": "pipeline"
    },
    6: {
        "name": "Materialize Splits",
        "script": "06_materialize_splits.py",
        "swap_after": False,
        "description": "Materialize CPCV splits into split_X_input collections",
        "stage_type": "pipeline"
    },
    7: {
        "name": "Stylized Facts - Raw Features",
        "script": "09_test_stylized_facts.py",
        "swap_after": False,
        "description": "Test stylized facts on raw materialized features (baseline)",
        "stage_type": "analysis",
        "depends_on": [6],
        "output_dir": "artifacts/stylized_facts/01_raw"
    },
    8: {
        "name": "Select Feature Transformations",
        "script": "07_feature_transform.py",
        "swap_after": False,
        "description": "Select optimal normalization transformations using validation data",
        "stage_type": "pipeline"
    },
    9: {
        "name": "Apply Feature Transformations",
        "script": "08_apply_feature_transforms.py",
        "swap_after": False,
        "description": "Apply selected transformations and rename: split_X_output -> split_X_input",
        "stage_type": "pipeline"
    },
    10: {
        "name": "Stylized Facts - Transformed Features",
        "script": "09_test_stylized_facts.py",
        "swap_after": False,
        "description": "Test stylized facts after transformations (post-normalization)",
        "stage_type": "analysis",
        "depends_on": [9],
        "output_dir": "artifacts/stylized_facts/02_transformed"
    },
    11: {
        "name": "Select EWMA Half-Lives",
        "script": "10_feature_scale.py",
        "swap_after": False,
        "description": "Select optimal EWMA half-life parameters for feature standardization",
        "stage_type": "pipeline"
    },
    12: {
        "name": "Apply EWMA Standardization",
        "script": "11_apply_feature_standardization.py",
        "swap_after": False,
        "description": "Apply EWMA standardization and rename: split_X_output -> split_X_input",
        "stage_type": "pipeline"
    },
    13: {
        "name": "Stylized Facts - Standardized Features",
        "script": "09_test_stylized_facts.py",
        "swap_after": False,
        "description": "Test stylized facts after EWMA standardization (final)",
        "stage_type": "analysis",
        "depends_on": [12],
        "output_dir": "artifacts/stylized_facts/03_standardized"
    },
    14: {
        "name": "Null Value Filtering",
        "script": "12_filter_nulls.py",
        "swap_after": False,
        "description": "Apply Null Filtering and rename: split_X_output -> split_X_input",
        "stage_type": "pipeline"
    }
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def is_stylized_facts_stage(stage_num):
    """Check if stage is a stylized facts analysis stage."""
    stage_info = STAGES.get(stage_num)
    return stage_info and stage_info.get("stage_type") == "analysis"

def should_skip_stage(stage_num):
    """Determine if stage should be skipped based on configuration."""
    if not CONFIG.get('enable_stylized_facts', True):
        if is_stylized_facts_stage(stage_num):
            return True
    return False

def get_stages_in_range(start, stop):
    """Get list of stages between start and stop (inclusive)."""
    stages = []
    for stage_num in range(start, stop + 1):
        if stage_num in STAGES and not should_skip_stage(stage_num):
            stages.append(stage_num)
    return stages

def run_stage_with_swap(pipeline_mgr: CyclicPipelineManager, 
                        stage_runner: StageRunner,
                        stage_num: int) -> bool:
    """
    Run a single stage and swap if needed.
    
    Args:
        pipeline_mgr: Pipeline manager instance
        stage_runner: Stage runner instance
        stage_num: Stage number to run
        
    Returns:
        True if stage succeeded, False otherwise
    """
    stage_info = STAGES[stage_num]
    
    log_section(f"Stage {stage_num}: {stage_info['name']}")
    logger(f"Description: {stage_info['description']}", "INFO")
    logger(f"Type: {stage_info['stage_type']}", "INFO")
    
    # Show output directory for analysis stages
    if stage_info.get("output_dir"):
        output_path = os.path.join(REPO_ROOT, stage_info["output_dir"])
        logger(f"Output directory: {output_path}", "INFO")
    
    # Check dependencies for analysis stages
    if stage_info.get("depends_on"):
        deps = stage_info["depends_on"]
        logger(f"Dependencies: {deps}", "DEBUG")
    
    # Validate prerequisites (skip for stages 6+ as they work with split collections)
    if stage_num <= 5:
        try:
            pipeline_mgr.validate_can_run_stage(stage_num)
        except ValueError as e:
            logger(f"Cannot run stage: {str(e)}", "ERROR")
            return False
    
    # Run stage
    logger(f"Executing {stage_info['script']}...", "INFO")
    
    # For analysis stages with output_dir, pass it as keyword argument
    if stage_info.get("output_dir"):
        # Convert output-dir to output_dir for kwargs (hyphens to underscores)
        result = stage_runner.run_stage(
            stage_info['script'],
            **{'output-dir': stage_info["output_dir"]}
        )
    else:
        result = stage_runner.run_stage(stage_info['script'])
    
    if not result["success"]:
        logger(f"Stage {stage_num} failed!", "ERROR")
        logger(f"Return code: {result['return_code']}", "ERROR")
        
        # For analysis stages, failure is non-fatal (warning only)
        if is_stylized_facts_stage(stage_num):
            logger(f"[WARNING]️  Analysis stage failed, but continuing pipeline...", "WARNING")
            return True  # Don't stop pipeline for analysis failures
        
        return False
    
    logger(f"Stage {stage_num} completed successfully in {result['duration']:.2f}s", "INFO")
    
    # Swap if needed (only for stages 3-5 with cyclic lob_input/lob_output pattern)
    if stage_info["swap_after"]:
        log_section(f"Swapping Collections After Stage {stage_num}")
        try:
            pipeline_mgr.swap_working_collections()
        except Exception as e:
            logger(f"Swap failed: {str(e)}", "ERROR")
            return False
    
    log_section("", char="=")
    
    return True

# =================================================================================================
# Main Function
# =================================================================================================

def main():
    """Main orchestrator function."""
    start_time = datetime.now()
    
    # Validate stage range
    if CONFIG['start_from'] > CONFIG['stop_at']:
        logger("Error: start_from must be <= stop_at", "ERROR")
        return 1
    
    if not (3 <= CONFIG['start_from'] <= 14 and 3 <= CONFIG['stop_at'] <= 14):
        logger("Error: stages must be between 3 and 14", "ERROR")
        return 1
    
    # Check if requested stages exist
    stages_to_run = get_stages_in_range(CONFIG['start_from'], CONFIG['stop_at'])
    if not stages_to_run:
        logger(f"No valid stages found in range {CONFIG['start_from']}-{CONFIG['stop_at']}", "ERROR")
        logger(f"Available stages: {sorted(STAGES.keys())}", "INFO")
        return 1
    
    log_section("Pipeline Orchestrator")
    logger(f"MongoDB URI: {CONFIG['mongo_uri']}", "INFO")
    logger(f"Database: {CONFIG['db_name']}", "INFO")
    logger(f"Stage range: {CONFIG['start_from']} to {CONFIG['stop_at']}", "INFO")
    logger(f"Stages to run: {stages_to_run}", "INFO")
    logger(f"Reset mode: {CONFIG['reset']}", "INFO")
    logger(f"Skip init: {CONFIG['skip_init']}", "INFO")
    logger(f"Stylized facts enabled: {CONFIG['enable_stylized_facts']}", "INFO")
    log_section("", char="=")
    
    try:
        # Create managers
        pipeline_mgr = CyclicPipelineManager(
            mongo_uri=CONFIG['mongo_uri'],
            db_name=CONFIG['db_name']
        )
        stage_runner = StageRunner(scripts_dir=SCRIPT_DIR)
        
        # Show initial state (only for stages 3-5)
        if CONFIG['start_from'] <= 5:
            log_section("Initial Pipeline State")
            pipeline_mgr.print_pipeline_state()
        
        # Reset if requested
        if CONFIG['reset'] and CONFIG['start_from'] <= 5:
            log_section("Resetting Pipeline")
            logger("Resetting to archive state...", "INFO")
            pipeline_mgr.reset_to_archive(force=True)
        
        # Initialize if needed
        if not CONFIG['skip_init'] and not CONFIG['reset'] and CONFIG['start_from'] <= 5:
            log_section("Checking Initialization")
            state = pipeline_mgr.get_pipeline_state()
            
            if not state["collections"]["lob_input"]["exists"]:
                logger("Pipeline not initialized, initializing now...", "INFO")
                pipeline_mgr.initialize_pipeline(force=False)
            elif state["collections"]["lob_input"]["count"] == 0:
                logger("lob_input is empty, re-initializing...", "WARNING")
                pipeline_mgr.initialize_pipeline(force=True)
            else:
                logger("Pipeline already initialized", "INFO")
        
        # Run stages
        logger(f"Running {len(stages_to_run)} stages: {stages_to_run}", "INFO")
        
        for stage_num in stages_to_run:
            success = run_stage_with_swap(pipeline_mgr, stage_runner, stage_num)
            
            if not success:
                logger(f"Pipeline stopped at Stage {stage_num} due to failure", "ERROR")
                stage_runner.print_execution_summary()
                return 1
        
        # Show final state (only for stages 3-5)
        if CONFIG['stop_at'] <= 5:
            log_section("Final Pipeline State")
            pipeline_mgr.print_pipeline_state()
        
        # Show execution summary
        stage_runner.print_execution_summary()
        
        # Calculate total duration
        duration = (datetime.now() - start_time).total_seconds()
        
        log_section("Pipeline Complete")
        logger(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)", "INFO")
        logger(f"Stages completed: {CONFIG['start_from']} to {CONFIG['stop_at']}", "INFO")
        logger(f"Successful stages: {len(stages_to_run)}", "INFO")
        log_section("", char="=")
        
        return 0
        
    except Exception as e:
        logger(f"Pipeline failed: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)