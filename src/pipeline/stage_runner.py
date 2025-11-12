import subprocess
import sys
import time
from typing import Optional, Dict, Any
from datetime import datetime

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import logger

env = os.environ.copy()
env['PIPELINE_ORCHESTRATED'] = 'true'

class StageRunner:
    """
    Executes individual pipeline stages.
    
    Responsibilities:
    - Run stage scripts as subprocesses
    - Capture and log output
    - Track execution time
    - Verify stage completion
    - Handle errors gracefully
    """
    
    def __init__(self, scripts_dir: str = "scripts"):
        """
        Initialize stage runner.
        
        Args:
            scripts_dir: Directory containing stage scripts
        """
        self.scripts_dir = scripts_dir
        self.execution_history = []
    
    def run_stage(self, script_name: str, python_executable: str = "python",
                  **kwargs) -> Dict[str, Any]:
        """
        Run a pipeline stage script.
        
        Args:
            script_name: Name of script to run (e.g., "03_data_splitting.py")
            python_executable: Python executable to use
            **kwargs: Additional command-line arguments for the script
            
        Returns:
            Dictionary with execution results:
            - success: Whether execution succeeded
            - duration: Execution time in seconds
            - return_code: Process return code
            - stdout: Standard output
            - stderr: Standard error
            
        Raises:
            FileNotFoundError: If script doesn't exist
        """
        script_path = os.path.join(self.scripts_dir, script_name)
        
        # Verify script exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        logger("=" * 80, "INFO")
        logger(f"RUNNING STAGE: {script_name}", "INFO")
        logger("=" * 80, "INFO")
        
        # Build command
        cmd = [python_executable, script_path]
        
        # Add any additional arguments
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger(f"Command: {' '.join(cmd)}", "INFO")
        
        # Execute
        start_time = time.time()
        start_timestamp = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero return
                env=env
            )
            
            duration = time.time() - start_time
            success = (result.returncode == 0)
            
            # Log results
            if success:
                logger(f"[OK] Stage completed successfully in {duration:.2f}s", "INFO")
            else:
                logger(f"[FAIL] Stage failed with return code {result.returncode}", "ERROR")
                logger(f"Duration: {duration:.2f}s", "ERROR")
            
            # Log stdout if present
            if result.stdout:
                logger("Stage output:", "INFO")
                print(result.stdout)
            
            # Log stderr if present
            if result.stderr:
                logger("Stage errors:", "ERROR")
                print(result.stderr, file=sys.stderr)
            
            # Build execution record
            execution_record = {
                "script_name": script_name,
                "start_time": start_timestamp,
                "duration": duration,
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": ' '.join(cmd)
            }
            
            self.execution_history.append(execution_record)
            
            logger("=" * 80, "INFO")
            
            return execution_record
            
        except Exception as e:
            duration = time.time() - start_time
            logger(f"[FAIL] Stage execution failed with exception: {str(e)}", "ERROR")
            
            execution_record = {
                "script_name": script_name,
                "start_time": start_timestamp,
                "duration": duration,
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "command": ' '.join(cmd)
            }
            
            self.execution_history.append(execution_record)
            
            logger("=" * 80, "INFO")
            
            return execution_record
    
    def get_execution_summary(self) -> str:
        """
        Get summary of all executed stages.
        
        Returns:
            Formatted string with execution history
        """
        if not self.execution_history:
            return "No stages executed yet"
        
        summary = []
        summary.append("=" * 80)
        summary.append("EXECUTION HISTORY")
        summary.append("=" * 80)
        
        for i, record in enumerate(self.execution_history, 1):
            status = "[OK] SUCCESS" if record["success"] else "[FAIL] FAILED"
            summary.append(f"{i}. {record['script_name']}")
            summary.append(f"   Status: {status}")
            summary.append(f"   Duration: {record['duration']:.2f}s")
            summary.append(f"   Return Code: {record['return_code']}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def print_execution_summary(self):
        """Print execution summary to console."""
        print(self.get_execution_summary())