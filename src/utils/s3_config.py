"""
S3 Configuration Utilities for Large Data Handling

This module provides optimized S3 configurations for handling large data volumes
(100MB parquet files and 10GB splits) with proper retry logic, timeouts, and
multipart upload/download settings.
"""

from typing import Dict, Any
from pyspark.sql import SparkSession
import boto3
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from botocore.exceptions import ClientError


# =================================================================================================
# S3 Configuration Constants
# =================================================================================================

# S3A Hadoop configuration for large file handling
S3A_LARGE_FILE_CONFIG = {
    # Multipart upload settings (critical for files >100MB)
    "fs.s3a.multipart.size": "104857600",  # 100MB per part
    "fs.s3a.multipart.threshold": "52428800",  # Start multipart at 50MB
    "fs.s3a.fast.upload": "true",
    "fs.s3a.fast.upload.buffer": "disk",  # Use disk buffering for large files (prevents OOM)

    # Connection pool settings
    "fs.s3a.connection.maximum": "50",  # Max connections
    "fs.s3a.threads.max": "20",  # Max threads for parallel uploads

    # Timeout settings (critical to prevent hanging)
    "fs.s3a.connection.establish.timeout": "30000",  # 30s to establish connection
    "fs.s3a.connection.timeout": "300000",  # 5min for large uploads/downloads
    "fs.s3a.socket.recv.buffer": "262144",  # 256KB receive buffer
    "fs.s3a.socket.send.buffer": "262144",  # 256KB send buffer

    # Retry settings (critical for reliability)
    "fs.s3a.attempts.maximum": "10",  # Max retry attempts
    "fs.s3a.retry.limit": "7",  # Retry limit for throttling
    "fs.s3a.retry.interval": "500ms",  # Base retry interval

    # Read-ahead for large downloads
    "fs.s3a.readahead.range": "2097152",  # 2MB read-ahead
    "fs.s3a.input.fadvise": "sequential",  # Sequential read optimization for large files

    # Performance tuning
    "fs.s3a.block.size": "134217728",  # 128MB block size
    "fs.s3a.committer.threads": "8",  # Threads for commit operations
}


# =================================================================================================
# Boto3 Configuration
# =================================================================================================

def get_boto3_config() -> Config:
    """
    Get optimized boto3 configuration for large data operations.

    Returns:
        Boto3 Config object with retry and timeout settings
    """
    return Config(
        retries={
            'max_attempts': 10,
            'mode': 'adaptive',  # Adaptive retry mode for better handling of throttling
        },
        connect_timeout=30,  # 30s connection timeout
        read_timeout=300,  # 5min read timeout for large files
        max_pool_connections=50,  # Connection pool size
        tcp_keepalive=True,  # Keep connections alive
    )


def create_s3_client(region_name: str = 'us-east-1') -> boto3.client:
    """
    Create S3 client with optimized configuration.

    Args:
        region_name: AWS region

    Returns:
        Configured boto3 S3 client
    """
    return boto3.client('s3', region_name=region_name, config=get_boto3_config())


# =================================================================================================
# Spark S3 Configuration
# =================================================================================================

def configure_spark_for_s3(
    spark: SparkSession,
    credentials_provider: str = "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    custom_config: Dict[str, str] = None
) -> None:
    """
    Configure Spark session for optimized S3 access with large files.

    Args:
        spark: Spark session to configure
        credentials_provider: AWS credentials provider class
        custom_config: Additional custom S3A configuration (overrides defaults)
    """
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

    # Set S3A file system implementation
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Set credentials provider
    hadoop_conf.set("fs.s3a.aws.credentials.provider", credentials_provider)

    # Apply large file configuration
    config = S3A_LARGE_FILE_CONFIG.copy()

    # Apply custom overrides if provided
    if custom_config:
        config.update(custom_config)

    # Set all configuration parameters
    for key, value in config.items():
        hadoop_conf.set(key, str(value))


# =================================================================================================
# Retry Decorators for S3 Operations
# =================================================================================================

def retry_s3_operation(func):
    """
    Decorator to add retry logic to S3 operations.

    Retries on network errors, throttling, and timeouts with exponential backoff.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((
            ClientError,
            ConnectionError,
            TimeoutError,
            OSError,  # Includes network-related OSErrors
        )),
        reraise=True
    )(func)


# =================================================================================================
# S3 Path Utilities
# =================================================================================================

def parse_s3_path(s3_path: str) -> tuple:
    """
    Parse S3 path into bucket and key.

    Args:
        s3_path: S3 path (s3://bucket/key or s3a://bucket/key)

    Returns:
        Tuple of (bucket, key)
    """
    # Remove s3:// or s3a:// prefix
    path = s3_path.replace("s3a://", "").replace("s3://", "")
    parts = path.split("/", 1)

    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    return bucket, key


def get_s3_object_size(s3_path: str, region: str = 'us-east-1') -> int:
    """
    Get size of S3 object without downloading.

    Args:
        s3_path: S3 path to object
        region: AWS region

    Returns:
        Size in bytes
    """
    bucket, key = parse_s3_path(s3_path)
    s3_client = create_s3_client(region)

    response = s3_client.head_object(Bucket=bucket, Key=key)
    return response['ContentLength']


def get_s3_etag(s3_path: str, region: str = 'us-east-1') -> str:
    """
    Get S3 ETag (checksum) without downloading file.

    This is much more efficient than downloading and checksumming large files.

    Args:
        s3_path: S3 path to object
        region: AWS region

    Returns:
        ETag string
    """
    bucket, key = parse_s3_path(s3_path)
    s3_client = create_s3_client(region)

    response = s3_client.head_object(Bucket=bucket, Key=key)
    return response.get('ETag', '').strip('"')


# =================================================================================================
# MongoDB Configuration for Large Imports
# =================================================================================================

MONGODB_LARGE_IMPORT_OPTIONS = {
    "maxBatchSize": "1000",  # Batch size for inserts
    "writeConcern.w": "1",  # Write concern level
    "ordered": "false",  # Unordered writes for better performance
}


def get_mongodb_read_options(partition_size_mb: int = 64) -> Dict[str, str]:
    """
    Get optimized MongoDB read options for large collections.

    Args:
        partition_size_mb: Size of each partition in MB

    Returns:
        Dictionary of MongoDB read options
    """
    return {
        "partitioner": "MongoSamplePartitioner",
        "partitionKey": "_id",
        "partitionSizeMB": str(partition_size_mb),
        "samplesPerPartition": "10",  # Samples for partitioning
    }


# =================================================================================================
# Monitoring Utilities
# =================================================================================================

class S3OperationMetrics:
    """Track metrics for S3 operations."""

    def __init__(self):
        self.upload_count = 0
        self.download_count = 0
        self.upload_bytes = 0
        self.download_bytes = 0
        self.retry_count = 0
        self.error_count = 0

    def record_upload(self, size_bytes: int):
        """Record an upload operation."""
        self.upload_count += 1
        self.upload_bytes += size_bytes

    def record_download(self, size_bytes: int):
        """Record a download operation."""
        self.download_count += 1
        self.download_bytes += size_bytes

    def record_retry(self):
        """Record a retry."""
        self.retry_count += 1

    def record_error(self):
        """Record an error."""
        self.error_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        return {
            "uploads": self.upload_count,
            "downloads": self.download_count,
            "upload_gb": round(self.upload_bytes / (1024**3), 2),
            "download_gb": round(self.download_bytes / (1024**3), 2),
            "retries": self.retry_count,
            "errors": self.error_count,
        }
