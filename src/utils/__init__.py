# Spark configuration
from .spark import create_spark_session

# Logging
from .logging import (
    logger,
    log_section,
    log_separator,
    log_progress,
    log_timing,
    get_timestamp,
    get_timestamp_utc,
    format_time_str
)

# Timestamp (naive UTC convention)
from .timestamp import (
    to_naive_utc,
    to_naive_datetime,
    parse_timestamp_from_metadata,
    format_timestamp_for_metadata,
    format_timestamp_for_mongodb,
    normalize_timestamp_string,
    get_millisecond_timestamp_string,
    parse_hour_string,
    format_hour_string,
    datetime_to_utc_iso,
    add_hours,
    get_hour_boundaries,
    extract_hour_from_timestamp_string,
    is_same_hour,
    utcnow,
    get_current_timestamp_string,
    get_current_hour_string,
    is_naive_utc,
    assert_naive_utc
)

# Database operations
from .database import (
    read_from_mongodb,
    read_sorted_with_timestamp_strings,
    write_to_mongodb,
    write_with_timestamp_conversion,
    update_log_collection,
    get_logged_files,
    count_documents
)

__all__ = [
    # Spark
    'create_spark_session',
    
    # Logging
    'logger',
    'log_section',
    'log_separator',
    'log_progress',
    'log_timing',
    'get_timestamp',
    'get_timestamp_utc',
    'format_time_str',
    
    # Timestamps (naive UTC)
    'to_naive_utc',
    'to_naive_datetime',
    'parse_timestamp_from_metadata',
    'format_timestamp_for_metadata',
    'format_timestamp_for_mongodb',
    'normalize_timestamp_string',
    'get_millisecond_timestamp_string',
    'parse_hour_string',
    'format_hour_string',
    'datetime_to_utc_iso',
    'add_hours',
    'get_hour_boundaries',
    'extract_hour_from_timestamp_string',
    'is_same_hour',
    'utcnow',
    'get_current_timestamp_string',
    'get_current_hour_string',
    'is_naive_utc',
    'assert_naive_utc',
    
    # Database
    'read_from_mongodb',
    'read_sorted_with_timestamp_strings',
    'write_to_mongodb',
    'write_with_timestamp_conversion',
    'update_log_collection',
    'get_logged_files',
    'count_documents',
]