import os
import sys
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger, log_section
from src.pipeline import CyclicPipelineManager

# =================================================================================================
# Configuration - Edit these settings
# =================================================================================================

CONFIG = {
    'mongo_uri': "mongodb://127.0.0.1:27017/",
    'db_name': "raw",
    'force': True,  # Set to True to drop existing working collections
}

# =================================================================================================
# Main Function
# =================================================================================================

def main():
    """Main initialization function."""
    start_time = datetime.now()
    
    log_section("Pipeline Initialization")
    logger(f"MongoDB URI: {CONFIG['mongo_uri']}", "INFO")
    logger(f"Database: {CONFIG['db_name']}", "INFO")
    logger(f"Force mode: {CONFIG['force']}", "INFO")
    log_section("", char="-")
    
    try:
        # Create pipeline manager
        manager = CyclicPipelineManager(
            mongo_uri=CONFIG['mongo_uri'],
            db_name=CONFIG['db_name']
        )
        
        # Show current state
        logger("Current pipeline state:", "INFO")
        manager.print_pipeline_state()
        
        # Initialize pipeline
        logger("Initializing pipeline...", "INFO")
        manager.initialize_pipeline(force=CONFIG['force'])
        
        # Show final state
        logger("Pipeline initialized successfully!", "INFO")
        manager.print_pipeline_state()
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        log_section("Initialization Complete")
        logger(f"Total time: {duration:.2f} seconds", "INFO")
        logger("Pipeline is ready for Stage 3 (Data Splitting)", "INFO")
        log_section("", char="=")
        
        return 0
        
    except Exception as e:
        logger(f"Initialization failed: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)