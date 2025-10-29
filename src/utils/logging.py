import sys
import time
from typing import Optional
from datetime import datetime, timezone

def logger(msg: str, level: str) -> None:
    """
    Logs a message with timestamp and level. This is the main logging function.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] [{level}] {msg}")
    sys.stdout.flush()

def log_separator(char: str = "=", length: int = 100) -> None:
    """
    Logs a visual separator line. Useful for separating sections in log output.
    """
    logger(char * length, "INFO")

def log_section(title: str, char: str = "=", length: int = 100) -> None:
    """
    Logs a section header with separators above and below. Creates a visually 
    distinct section in log output.
    """
    log_separator(char, length)
    logger(title, "INFO")
    log_separator(char, length)

def get_timestamp() -> str:
    """
    Returns current timestamp in ISO format (local time).
    """
    return datetime.now().isoformat()

def get_timestamp_utc() -> str:
    """
    Returns current UTC timestamp in ISO format with Z marker.
    """
    return datetime.now(timezone.utc).isoformat()


def format_time_str(timestamp: Optional[datetime] = None) -> str:
    """
    Formats a datetime for simple display (HH:MM:SS). Useful for concise time 
    display in logs.
    """
    if timestamp is None:
        timestamp = datetime.now()

    return timestamp.strftime('%H:%M:%S')

def log_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """
    Logs progress with percentage completion. Useful for batch processing loops.
    """
    percentage = (current / total) * 100
    logger(f"{prefix}: {current}/{total} ({percentage:.1f}%)", "INFO")

def log_timing(operation: str, start_time: float, end_time: Optional[float] = None) -> None:
    """
    Logs the duration of an operation. Useful for performance monitoring.
    """
    if end_time is None:
        end_time = time.time()

    duration = end_time - start_time
    logger(f"{operation} completed in {duration:.2f}s", "INFO")