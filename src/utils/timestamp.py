from typing import Union, Optional
from datetime import datetime, timezone, timedelta

# =================================================================================================
# Core Conversion Functions
# =================================================================================================

def to_naive_utc(dt: datetime) -> datetime:
    """
    Converts any datetime object to naive UTC datetime.
    """
    if dt is None:
        return None
    
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Convert to UTC first if not already, then remove timezone
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None)
    
    # Already naive - assume it's UTC
    return dt

def to_naive_datetime(dt: datetime) -> datetime:
    """
    Alias for to_naive_utc() for backwards compatibility.
    """
    return to_naive_utc(dt)

# =================================================================================================
# Parsing Functions (String → Naive UTC Datetime)
# =================================================================================================

def parse_timestamp_from_metadata(ts: Union[str, datetime]) -> datetime:
    """
    Parses timestamp from metadata YAML file to naive UTC datetime.
    """
    if isinstance(ts, datetime):
        return to_naive_utc(ts)
    
    # Remove timezone markers
    ts_clean = ts.replace('Z', '').replace('+00:00', '')
    
    return datetime.fromisoformat(ts_clean)

def normalize_timestamp_string(ts_str: str) -> datetime:
    """
    Normalizes timestamp string to naive UTC datetime for consistent comparison.
    """
    # Remove timezone markers
    ts_clean = ts_str.replace('Z', '').replace('+00:00', '')
    
    return datetime.fromisoformat(ts_clean)

def parse_hour_string(hour_str: str) -> datetime:
    """
    Parses an hour string to naive UTC datetime.
    """
    return normalize_timestamp_string(hour_str)

# =================================================================================================
# Formatting Functions (Naive UTC Datetime → String)
# =================================================================================================

def format_timestamp_for_metadata(dt: datetime) -> str:
    """
    Formats naive UTC datetime for storage in metadata YAML files.
    """
    return dt.isoformat() + 'Z'

def format_timestamp_for_mongodb(dt: datetime) -> str:
    """
    Formats naive UTC datetime for MongoDB queries.
    """
    return dt.isoformat() + 'Z'

def get_millisecond_timestamp_string(dt: Optional[datetime] = None) -> str:
    """
    Returns timestamp string with millisecond precision in standard format.
    """
    if dt is None:
        dt = datetime.utcnow()  # Get current UTC time as naive datetime
    
    # Format with milliseconds (3 digits) - strftime %f gives 6 digits, take first 3
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def format_hour_string(dt: datetime) -> str:
    """
    Formats a naive UTC datetime as an hour string for batch processing.
    """
    hour_dt = dt.replace(minute=0, second=0, microsecond=0)
    
    return get_millisecond_timestamp_string(hour_dt)

def datetime_to_utc_iso(dt: Optional[datetime] = None) -> str:
    """
    Converts datetime to UTC ISO format with Z marker.
    """
    if dt is None:
        dt = datetime.utcnow()  # Naive UTC
    
    if dt.tzinfo is None:
        # Naive datetime - assume it's already in UTC
        return dt.isoformat() + 'Z'
    else:
        # Timezone-aware - convert to UTC first
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None).isoformat() + 'Z'

# =================================================================================================
# Time Arithmetic & Utilities
# =================================================================================================

def add_hours(dt: datetime, hours: int) -> datetime:
    """
    Adds hours to a naive UTC datetime.
    """
    return dt + timedelta(hours=hours)

def get_hour_boundaries(dt: datetime) -> tuple[datetime, datetime]:
    """
    Gets the start and end boundaries of the hour containing the given datetime.
    """
    hour_start = dt.replace(minute=0, second=0, microsecond=0)
    hour_end = hour_start + timedelta(hours=1)
    
    return hour_start, hour_end

def extract_hour_from_timestamp_string(ts_str: str) -> str:
    """
    Extracts the hour string from a full timestamp string.
    """
    # Take first 13 characters (up to hour) and add standard suffix
    return ts_str[:13] + ':00:00.000Z'

def is_same_hour(dt1: datetime, dt2: datetime) -> bool:
    """
    Checks if two naive UTC datetimes are in the same hour.
    """
    return (dt1.year == dt2.year and 
            dt1.month == dt2.month and 
            dt1.day == dt2.day and 
            dt1.hour == dt2.hour)

# =================================================================================================
# Current Time Functions (Always Return Naive UTC)
# =================================================================================================

def utcnow() -> datetime:
    """
    Returns current UTC time as naive datetime.
    """
    return datetime.utcnow()

def get_current_timestamp_string() -> str:
    """
    Returns current UTC time as standard timestamp string.
    """
    return get_millisecond_timestamp_string(utcnow())

def get_current_hour_string() -> str:
    """
    Returns current UTC hour as standard hour string.
    """
    return format_hour_string(utcnow())

# =================================================================================================
# Validation Functions
# =================================================================================================

def is_naive_utc(dt: datetime) -> bool:
    """
    Checks if a datetime is naive (no timezone info).
    """
    return dt.tzinfo is None

def assert_naive_utc(dt: datetime, msg: str = "Datetime must be naive UTC") -> None:
    """
    Asserts that a datetime is naive UTC.
    """
    assert is_naive_utc(dt), f"{msg}: {dt}"