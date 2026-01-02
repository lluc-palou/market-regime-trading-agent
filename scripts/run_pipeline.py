import os
import sys
from datetime import datetime
from pathlib import Path

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

    # Pipeline mode
    'mode': 'train',  # 'train' or 'test'

    # Pipeline control
    'start_from': 2,       # Start from stage 2-21
    'stop_at': 21,         # Stop at stage 2-21

    # Stylized facts testing (train mode only)
    'enable_stylized_facts': True,  # Enable stylized facts analysis stages

    # Test mode settings
    'test_split': 0,  # Which split to use for full training in test mode
}

# =================================================================================================
# Stage Definitions
# =================================================================================================

STAGES = {
    # =============================================================================================
    # Data Preparation Pipeline (Stages 2-6)
    # =============================================================================================
    2: {
        "name": "Data Ingestion",
        "script": "02_data_ingestion.py",
        "swap_after": True,
        "description": "Ingest raw LOB data from parquet files to MongoDB",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (already done)
    },
    3: {
        "name": "Data Splitting",
        "script": "03_data_splitting.py",
        "swap_after": True,
        "description": "Assign CPCV fold IDs to samples",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (already done)
    },
    4: {
        "name": "Feature Derivation",
        "script": "04_feature_derivation.py",
        "swap_after": True,
        "description": "Derive LOB-based features (microprice, depth, etc.)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (already done)
    },
    5: {
        "name": "LOB Standardization",
        "script": "05_lob_standardization.py",
        "swap_after": True,
        "description": "Standardize LOB features (price/spread normalization)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (already done)
    },
    6: {
        "name": "Materialize Splits",
        "script": "06_materialize_splits.py",
        "swap_after": False,
        "description": "Create split_X collections + test_data collection",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (already done)
    },

    # =============================================================================================
    # Feature Preprocessing Pipeline (Stages 7-14)
    # =============================================================================================
    7: {
        "name": "Stylized Facts - Raw",
        "script": "09_test_stylized_facts.py",
        "swap_after": False,
        "description": "Test stylized facts on raw materialized features",
        "stage_type": "analysis",
        "depends_on": [6],
        "output_dir": "artifacts/stylized_facts/01_raw",
        "train_mode": True,
        "test_mode": False,  # Skip in test (no analysis on test data)
    },
    8: {
        "name": "Select Feature Transformations",
        "script": "07_feature_transform.py",
        "swap_after": False,
        "description": "TRAIN: Fit transformers on validation data | TEST: Skip (use fitted)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (use fitted transformers)
    },
    9: {
        "name": "Apply Feature Transformations",
        "script": "08_apply_feature_transforms.py",
        "swap_after": False,
        "description": "TRAIN: Apply to split_X | TEST: Apply to test_data",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": True,  # Apply fitted transformers to test_data
        "pass_mode": True,  # Pass --mode argument to script
    },
    10: {
        "name": "Stylized Facts - Transformed",
        "script": "09_test_stylized_facts.py",
        "swap_after": False,
        "description": "Test stylized facts after transformations",
        "stage_type": "analysis",
        "depends_on": [9],
        "output_dir": "artifacts/stylized_facts/02_transformed",
        "train_mode": True,
        "test_mode": False,  # Skip in test
    },
    11: {
        "name": "Select EWMA Half-Lives",
        "script": "10_feature_scale.py",
        "swap_after": False,
        "description": "TRAIN: Fit EWMA scalers on validation data | TEST: Skip (use fitted)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (use fitted scalers)
    },
    12: {
        "name": "Apply EWMA Standardization",
        "script": "11_apply_feature_standardization.py",
        "swap_after": False,
        "description": "TRAIN: Apply to split_X | TEST: Apply to test_data",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": True,  # Apply fitted scalers to test_data
        "pass_mode": True,  # Pass --mode argument to script
    },
    13: {
        "name": "Stylized Facts - Standardized",
        "script": "09_test_stylized_facts.py",
        "swap_after": False,
        "description": "Test stylized facts after EWMA standardization",
        "stage_type": "analysis",
        "depends_on": [12],
        "output_dir": "artifacts/stylized_facts/03_standardized",
        "train_mode": True,
        "test_mode": False,  # Skip in test
    },
    14: {
        "name": "Null Filtering",
        "script": "12_filter_nulls.py",
        "swap_after": False,
        "description": "TRAIN: Filter split_X | TEST: Filter test_data",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": True,  # Filter test_data
        "pass_mode": True,  # Pass --mode argument to script
    },

    # =============================================================================================
    # Model Training & Validation Pipeline (Stages 15-21)
    # =============================================================================================
    15: {
        "name": "VQ-VAE Hyperparameter Search",
        "script": "13_vqvae_hyperparameter_search.py",
        "swap_after": False,
        "description": "TRAIN: Search best VQ-VAE config | TEST: Skip (use best config)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (use best config from train)
    },
    16: {
        "name": "VQ-VAE Production",
        "script": "14_vqvae_production.py",
        "swap_after": False,
        "description": "TRAIN: Train per split | TEST: Train on full split_0 + encode test_data",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": True,  # Train on full split, encode test_data
        "pass_mode": True,  # Pass --mode argument to script
    },
    17: {
        "name": "Prior Hyperparameter Search",
        "script": "15_prior_hyperparameter_search.py",
        "swap_after": False,
        "description": "TRAIN: Search best Prior config | TEST: Skip (not needed)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (not needed)
    },
    18: {
        "name": "Prior Production",
        "script": "16_prior_production.py",
        "swap_after": False,
        "description": "TRAIN: Train per split | TEST: Skip (not needed)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (not needed)
    },
    19: {
        "name": "Synthetic Generation",
        "script": "17_synthetic_generation.py",
        "swap_after": False,
        "description": "TRAIN: Generate synthetic data per split | TEST: Skip (not needed)",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": False,  # Skip in test (not needed)
    },
    20: {
        "name": "Quality Assurance",
        "script": "19_generalization_validation.py",
        "swap_after": False,
        "description": "TRAIN: Validate VQ-VAE/Prior quality | TEST: Skip (QA done in train)",
        "stage_type": "analysis",
        "train_mode": True,
        "test_mode": False,  # Skip in test
    },
    21: {
        "name": "PPO Training/Evaluation",
        "script": "18_ppo_training.py",
        "swap_after": False,
        "description": "TRAIN: CPCV train/val | TEST: Train on full split_0 + evaluate on test_data",
        "stage_type": "pipeline",
        "train_mode": True,
        "test_mode": True,  # Train on full split, evaluate on test_data
        "pass_mode": True,  # Pass --mode argument to script
    },
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def is_analysis_stage(stage_num):
    """Check if stage is an analysis stage (non-fatal failures)."""
    stage_info = STAGES.get(stage_num)
    return stage_info and stage_info.get("stage_type") == "analysis"

def should_skip_stage(stage_num, mode):
    """
    Determine if stage should be skipped based on mode and configuration.

    Args:
        stage_num: Stage number
        mode: 'train' or 'test'

    Returns:
        True if stage should be skipped, False otherwise
    """
    stage_info = STAGES.get(stage_num)
    if not stage_info:
        return True

    # Check if stage is enabled for current mode
    if mode == 'train' and not stage_info.get('train_mode', True):
        return True
    if mode == 'test' and not stage_info.get('test_mode', False):
        return True

    # Check stylized facts configuration (train mode only)
    if mode == 'train' and not CONFIG.get('enable_stylized_facts', True):
        if is_analysis_stage(stage_num):
            return True

    return False

def validate_test_mode_artifacts():
    """
    Validate that required artifacts from train mode exist before running test mode.

    Returns:
        True if all required artifacts exist, False otherwise
    """
    artifact_base = Path(REPO_ROOT) / "artifacts"
    required_artifacts = [
        artifact_base / "feature_transform" / "best_transformations.json",
        artifact_base / "feature_scale" / "best_half_lives.json",
        artifact_base / "vqvae_models" / "hyperparameter_search" / "best_config.json",
    ]

    missing = []
    for artifact in required_artifacts:
        if not artifact.exists():
            missing.append(str(artifact))

    if missing:
        logger("ERROR: Test mode requires artifacts from train mode!", "ERROR")
        logger("Missing artifacts:", "ERROR")
        for artifact in missing:
            logger(f"  - {artifact}", "ERROR")
        logger("", "INFO")
        logger("Please run train mode first (--mode train) to generate required artifacts", "INFO")
        return False

    # Check if test_data collection exists
    from pymongo import MongoClient
    client = MongoClient(CONFIG['mongo_uri'])
    db = client[CONFIG['db_name']]

    if 'test_data' not in db.list_collection_names():
        logger("ERROR: test_data collection does not exist!", "ERROR")
        logger("Please run train mode through stage 6 (materialization) first", "INFO")
        client.close()
        return False

    client.close()
    return True

def get_stages_in_range(start, stop, mode):
    """
    Get list of stages between start and stop (inclusive) for given mode.

    Args:
        start: Starting stage number
        stop: Ending stage number
        mode: 'train' or 'test'

    Returns:
        List of stage numbers to run
    """
    stages = []
    for stage_num in range(start, stop + 1):
        if stage_num in STAGES and not should_skip_stage(stage_num, mode):
            stages.append(stage_num)
    return stages

def run_stage_with_swap(pipeline_mgr: CyclicPipelineManager,
                        stage_runner: StageRunner,
                        stage_num: int,
                        mode: str) -> bool:
    """
    Run a single stage and swap if needed.

    Args:
        pipeline_mgr: Pipeline manager instance
        stage_runner: Stage runner instance
        stage_num: Stage number to run
        mode: 'train' or 'test'

    Returns:
        True if stage succeeded, False otherwise
    """
    stage_info = STAGES[stage_num]

    log_section(f"Stage {stage_num}: {stage_info['name']}")
    logger(f"Mode: {mode.upper()}", "INFO")
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

    # Validate prerequisites (skip for stage 2 and stages 6+ as they work differently)
    # Stage 2: reads from files, not collections
    # Stages 6+: work with split collections
    if mode == 'train' and 3 <= stage_num <= 5:
        try:
            pipeline_mgr.validate_can_run_stage(stage_num)
        except ValueError as e:
            logger(f"Cannot run stage: {str(e)}", "ERROR")
            return False

    # Run stage
    logger(f"Executing {stage_info['script']}...", "INFO")

    # Build kwargs for stage execution
    kwargs = {}

    # Add output directory for analysis stages
    if stage_info.get("output_dir"):
        kwargs['output-dir'] = stage_info["output_dir"]

    # Add mode parameter if stage needs it
    if stage_info.get("pass_mode", False):
        kwargs['mode'] = mode

    # Add test_split parameter for test mode
    if mode == 'test' and stage_info.get("pass_mode", False):
        kwargs['test-split'] = str(CONFIG['test_split'])

    # Execute stage
    result = stage_runner.run_stage(stage_info['script'], **kwargs)

    if not result["success"]:
        logger(f"Stage {stage_num} failed!", "ERROR")
        logger(f"Return code: {result['return_code']}", "ERROR")

        # For analysis stages, failure is non-fatal (warning only)
        if is_analysis_stage(stage_num):
            logger(f"[WARNING] Analysis stage failed, but continuing pipeline...", "WARNING")
            return True  # Don't stop pipeline for analysis failures

        return False

    logger(f"Stage {stage_num} completed successfully in {result['duration']:.2f}s", "INFO")

    # Swap if needed (for stages with cyclic input/output pattern)
    # Only relevant for train mode stages 2-5
    if mode == 'train' and stage_info["swap_after"]:
        log_section(f"Swapping Collections After Stage {stage_num}")
        try:
            # Stage 2 is special: it's the first stage, so input doesn't exist yet
            # Just rename output -> input (no drop needed)
            if stage_num == 2:
                from pymongo import MongoClient
                client = MongoClient(CONFIG['mongo_uri'])
                db = client[CONFIG['db_name']]

                # Drop old input if it exists (cleanup from previous runs)
                if 'input' in db.list_collection_names():
                    db['input'].drop()
                    logger("Dropped existing 'input' collection", "INFO")

                # Rename output -> input
                if 'output' in db.list_collection_names():
                    db['output'].rename('input')
                    logger("Renamed 'output' -> 'input'", "INFO")
                else:
                    raise ValueError("Output collection 'output' does not exist!")

                client.close()
            else:
                # Stages 3-5: normal swap (drop input, rename output -> input)
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

    # Get mode from config
    mode = CONFIG['mode'].lower()
    if mode not in ['train', 'test']:
        logger(f"Error: Invalid mode '{mode}'. Must be 'train' or 'test'", "ERROR")
        return 1

    # Validate test mode artifacts
    if mode == 'test':
        if not validate_test_mode_artifacts():
            return 1

    # Validate stage range
    if CONFIG['start_from'] > CONFIG['stop_at']:
        logger("Error: start_from must be <= stop_at", "ERROR")
        return 1

    if not (2 <= CONFIG['start_from'] <= 21 and 2 <= CONFIG['stop_at'] <= 21):
        logger("Error: stages must be between 2 and 21", "ERROR")
        return 1

    # Check if requested stages exist
    stages_to_run = get_stages_in_range(CONFIG['start_from'], CONFIG['stop_at'], mode)
    if not stages_to_run:
        logger(f"No valid stages found in range {CONFIG['start_from']}-{CONFIG['stop_at']} for mode '{mode}'", "ERROR")
        logger(f"Available stages: {sorted(STAGES.keys())}", "INFO")
        return 1

    # Show pipeline configuration
    log_section("Pipeline Orchestrator")
    logger(f"Mode: {mode.upper()}", "INFO")
    logger(f"MongoDB URI: {CONFIG['mongo_uri']}", "INFO")
    logger(f"Database: {CONFIG['db_name']}", "INFO")
    logger(f"Stage range: {CONFIG['start_from']} to {CONFIG['stop_at']}", "INFO")
    logger(f"Stages to run: {stages_to_run}", "INFO")

    if mode == 'train':
        logger(f"Stylized facts enabled: {CONFIG['enable_stylized_facts']}", "INFO")
    else:  # test mode
        logger(f"Test split: {CONFIG['test_split']}", "INFO")

    log_section("", char="=")

    try:
        # Create managers
        pipeline_mgr = CyclicPipelineManager(
            mongo_uri=CONFIG['mongo_uri'],
            db_name=CONFIG['db_name']
        )
        stage_runner = StageRunner(scripts_dir=SCRIPT_DIR)

        # Run stages
        logger(f"Running {len(stages_to_run)} stages in {mode} mode: {stages_to_run}", "INFO")

        for stage_num in stages_to_run:
            success = run_stage_with_swap(pipeline_mgr, stage_runner, stage_num, mode)

            if not success:
                logger(f"Pipeline stopped at Stage {stage_num} due to failure", "ERROR")
                stage_runner.print_execution_summary()
                return 1

        # Show execution summary
        stage_runner.print_execution_summary()

        # Calculate total duration
        duration = (datetime.now() - start_time).total_seconds()

        log_section("Pipeline Complete")
        logger(f"Mode: {mode.upper()}", "INFO")
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
