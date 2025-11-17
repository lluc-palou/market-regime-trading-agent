"""
Upload LOB Data to S3

Uploads raw .parquet files from the local lob_data directory to S3 for storage and backup.
This is separate from the processed splits (stages 15-16) and uses a different S3 prefix.

Workflow:
1. Scans lob_data directory for .parquet files
2. Uploads each file to S3 with preserved filename
3. Generates manifest.json with metadata (file names, sizes, checksums)
4. Enables data versioning via run_id timestamps

Usage:
    python scripts/upload_lob_data_to_s3.py
    python scripts/upload_lob_data_to_s3.py --run-id custom_name
    python scripts/upload_lob_data_to_s3.py --source-dir path/to/custom/dir
"""

import sys
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# CRITICAL: Windows UTF-8 Fix - MUST BE BEFORE OTHER IMPORTS!
# =================================================================================================
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
# =================================================================================================

from pyspark.sql import SparkSession

from src.utils.logging import logger, log_section

# =================================================================================================
# AWS S3 Configuration
# =================================================================================================

S3_CONFIG = {
    "bucket": "lluc-tfg-data",           # S3 bucket name
    "prefix": "lob-data",                      # Prefix for LOB data (different from processed-splits)
    "region": "us-east-1",                     # AWS region
    # IAM role will be used for authentication (no keys needed in code)
}

# =================================================================================================
# Local Data Configuration
# =================================================================================================

LOCAL_CONFIG = {
    "data_dir": "lob_data",                    # Default local directory containing .parquet files
    "file_extension": ".parquet",              # File extension to look for
}

# =================================================================================================
# Spark Configuration
# =================================================================================================

SPARK_CONFIG = {
    "app_name": "UploadLobDataToS3",
    "driver_memory": "4g",
    "jar_files_path": "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/",
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def discover_parquet_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Discover all .parquet files in the specified directory.

    Args:
        data_dir: Path to directory containing .parquet files

    Returns:
        List of dictionaries with file information
    """
    logger(f'Scanning directory for .parquet files: {data_dir}', "INFO")

    if not os.path.exists(data_dir):
        logger(f'Directory does not exist: {data_dir}', "ERROR")
        logger('Please create the directory and add .parquet files before running this script', "ERROR")
        return []

    if not os.path.isdir(data_dir):
        logger(f'Path is not a directory: {data_dir}', "ERROR")
        return []

    parquet_files = []

    for file_path in Path(data_dir).rglob(f"*{LOCAL_CONFIG['file_extension']}"):
        file_info = {
            "filename": file_path.name,
            "relative_path": str(file_path.relative_to(data_dir)),
            "absolute_path": str(file_path.absolute()),
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        parquet_files.append(file_info)

    parquet_files.sort(key=lambda x: x["filename"])

    logger(f'Found {len(parquet_files)} .parquet files', "INFO")

    if parquet_files:
        total_size_mb = sum(f["size_mb"] for f in parquet_files)
        logger(f'Total size: {total_size_mb:.2f} MB', "INFO")

    return parquet_files


def calculate_file_checksum(file_path: str) -> str:
    """
    Calculate MD5 checksum for a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 checksum as hex string
    """
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        # Read file in 8MB chunks for better performance with large files
        for chunk in iter(lambda: f.read(8388608), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def upload_file_to_s3(
    spark: SparkSession,
    file_info: Dict[str, Any],
    s3_path: str
) -> Dict[str, Any]:
    """
    Upload a single .parquet file to S3 with retry logic.

    Args:
        spark: Spark session
        file_info: Dictionary with file information
        s3_path: Destination S3 path

    Returns:
        Dictionary with upload statistics
    """
    from src.utils.s3_config import retry_s3_operation

    logger(f'Uploading: {file_info["filename"]}', "INFO")
    logger(f'  Source: {file_info["absolute_path"]}', "INFO")
    logger(f'  Destination: {s3_path}', "INFO")
    logger(f'  Size: {file_info["size_mb"]:.2f} MB', "INFO")

    try:
        # Calculate checksum before upload
        checksum = calculate_file_checksum(file_info["absolute_path"])

        # Read local Parquet file
        df = spark.read.parquet(file_info["absolute_path"])

        # Get row count and schema info
        row_count = df.count()
        schema_fields = [field.name for field in df.schema.fields]

        # Write to S3 with retry logic for reliability
        @retry_s3_operation
        def write_to_s3():
            df.write.mode("overwrite").parquet(s3_path)

        write_to_s3()

        logger(f'  Uploaded successfully ({row_count:,} rows)', "INFO")

        return {
            "filename": file_info["filename"],
            "relative_path": file_info["relative_path"],
            "s3_path": s3_path,
            "size_bytes": file_info["size_bytes"],
            "size_mb": file_info["size_mb"],
            "row_count": row_count,
            "schema_fields": schema_fields,
            "checksum": f"md5:{checksum}",
            "upload_timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger(f'  Error uploading file: {str(e)}', "ERROR")
        return {
            "filename": file_info["filename"],
            "relative_path": file_info["relative_path"],
            "error": str(e),
            "upload_timestamp": datetime.utcnow().isoformat() + "Z",
        }


def generate_manifest(
    run_id: str,
    file_stats: List[Dict[str, Any]],
    source_dir: str
) -> Dict[str, Any]:
    """
    Generate manifest file with metadata about the upload.

    Args:
        run_id: Unique identifier for this upload run
        file_stats: List of file upload statistics
        source_dir: Source directory path

    Returns:
        Manifest dictionary
    """
    successful_uploads = [f for f in file_stats if "error" not in f]
    failed_uploads = [f for f in file_stats if "error" in f]

    return {
        "run_id": run_id,
        "upload_timestamp": datetime.utcnow().isoformat() + "Z",
        "source_directory": source_dir,
        "num_files": len(file_stats),
        "num_successful": len(successful_uploads),
        "num_failed": len(failed_uploads),
        "format": "parquet",
        "s3_config": {
            "bucket": S3_CONFIG["bucket"],
            "prefix": S3_CONFIG["prefix"],
            "region": S3_CONFIG["region"],
        },
        "files": file_stats,
        "total_size_mb": sum(f.get("size_mb", 0) for f in successful_uploads),
        "total_rows": sum(f.get("row_count", 0) for f in successful_uploads),
    }


def upload_manifest_to_s3(spark: SparkSession, manifest: Dict[str, Any], s3_manifest_path: str):
    """
    Upload manifest JSON to S3.

    Args:
        spark: Spark session
        manifest: Manifest dictionary
        s3_manifest_path: S3 path for manifest file
    """
    logger('Uploading manifest to S3...', "INFO")
    logger(f'  Destination: {s3_manifest_path}', "INFO")

    # Convert manifest to JSON string
    manifest_json = json.dumps(manifest, indent=2)

    # Create DataFrame with single row containing JSON
    manifest_df = spark.createDataFrame([(manifest_json,)], ["manifest_json"])

    # Write to S3 as JSON (single file)
    manifest_df.coalesce(1).write.mode("overwrite").json(s3_manifest_path)

    logger('Manifest uploaded successfully', "INFO")


def create_spark_session_for_s3(app_name: str, driver_memory: str, jar_files_path: str) -> SparkSession:
    """
    Create Spark session configured for S3 access with optimized settings for large files.

    Args:
        app_name: Name of the Spark application
        driver_memory: Driver memory allocation
        jar_files_path: Path to JAR files

    Returns:
        Configured Spark session
    """
    from src.utils.s3_config import configure_spark_for_s3

    # JAR files needed for S3 access
    aws_jars = [
        f"{jar_files_path}hadoop-aws-3.3.4.jar",
        f"{jar_files_path}aws-java-sdk-bundle-1.12.262.jar",
    ]

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.jars", ",".join(aws_jars))
        .getOrCreate()
    )

    # Configure S3 with optimized settings for large files (multipart, timeouts, retries)
    configure_spark_for_s3(spark)

    return spark


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Upload LOB data .parquet files to S3')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Custom run ID (default: timestamp)')
    parser.add_argument('--source-dir', type=str, default=None,
                        help=f'Source directory containing .parquet files (default: {LOCAL_CONFIG["data_dir"]})')
    args = parser.parse_args()

    # Generate run ID (timestamp or custom)
    run_id = args.run_id if args.run_id else datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Determine source directory
    source_dir = args.source_dir if args.source_dir else os.path.join(REPO_ROOT, LOCAL_CONFIG["data_dir"])

    log_section('UPLOAD LOB DATA TO S3')
    logger('', "INFO")

    logger('Configuration:', "INFO")
    logger(f'  Source Directory: {source_dir}', "INFO")
    logger(f'  S3 Bucket: {S3_CONFIG["bucket"]}', "INFO")
    logger(f'  S3 Prefix: {S3_CONFIG["prefix"]}', "INFO")
    logger(f'  S3 Region: {S3_CONFIG["region"]}', "INFO")
    logger(f'  Run ID: {run_id}', "INFO")
    logger('', "INFO")

    # Create Spark session with S3 support
    logger('Initializing Spark with S3 support...', "INFO")
    spark = create_spark_session_for_s3(
        app_name=SPARK_CONFIG["app_name"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        jar_files_path=SPARK_CONFIG["jar_files_path"]
    )
    logger('Spark session created with optimized S3 configuration', "INFO")
    logger('', "INFO")

    try:
        # =====================================================================
        # Step 1: Discover Parquet Files
        # =====================================================================

        log_section('STEP 1: DISCOVER PARQUET FILES')
        parquet_files = discover_parquet_files(source_dir)

        if not parquet_files:
            logger('No .parquet files found! Exiting.', "ERROR")
            logger('', "INFO")
            logger('To use this script:', "INFO")
            logger(f'  1. Create directory: {source_dir}', "INFO")
            logger(f'  2. Add .parquet files to the directory', "INFO")
            logger('  3. Run this script again', "INFO")
            return 1

        logger('', "INFO")

        # =====================================================================
        # Step 2: Upload Files to S3
        # =====================================================================

        log_section('STEP 2: UPLOAD FILES TO S3')

        file_stats = []

        for i, file_info in enumerate(parquet_files):
            logger(f'[{i+1}/{len(parquet_files)}] Processing {file_info["filename"]}...', "INFO")

            # Construct S3 path preserving relative path structure
            s3_path = f"s3a://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/{run_id}/{file_info['relative_path']}"

            stats = upload_file_to_s3(spark, file_info, s3_path)
            file_stats.append(stats)

            logger('', "INFO")

        # =====================================================================
        # Step 3: Generate and Upload Manifest
        # =====================================================================

        log_section('STEP 3: GENERATE MANIFEST')

        manifest = generate_manifest(run_id, file_stats, source_dir)

        logger('Manifest summary:', "INFO")
        logger(f'  Run ID: {manifest["run_id"]}', "INFO")
        logger(f'  Timestamp: {manifest["upload_timestamp"]}', "INFO")
        logger(f'  Total files: {manifest["num_files"]}', "INFO")
        logger(f'  Successful: {manifest["num_successful"]}', "INFO")
        logger(f'  Failed: {manifest["num_failed"]}', "INFO")
        logger(f'  Total size: {manifest["total_size_mb"]:.2f} MB', "INFO")
        logger(f'  Total rows: {manifest["total_rows"]:,}', "INFO")
        logger('', "INFO")

        # Upload manifest to S3
        s3_manifest_path = f"s3a://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/{run_id}/manifest.json"
        upload_manifest_to_s3(spark, manifest, s3_manifest_path)

        # Also save manifest locally for reference
        local_manifest_path = os.path.join(REPO_ROOT, "artifacts", "lob_data_uploads", f"manifest_{run_id}.json")
        os.makedirs(os.path.dirname(local_manifest_path), exist_ok=True)
        with open(local_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger(f'Manifest also saved locally: {local_manifest_path}', "INFO")

        logger('', "INFO")

        # =====================================================================
        # Completion
        # =====================================================================

        log_section('UPLOAD COMPLETE')

        if manifest["num_failed"] > 0:
            logger(f'Uploaded {manifest["num_successful"]} of {manifest["num_files"]} files to S3', "WARNING")
            logger(f'{manifest["num_failed"]} files failed - check logs above for details', "WARNING")
        else:
            logger(f'Successfully uploaded all {manifest["num_successful"]} files to S3', "INFO")

        logger(f'S3 Location: s3://{S3_CONFIG["bucket"]}/{S3_CONFIG["prefix"]}/{run_id}/', "INFO")
        logger(f'Run ID: {run_id}', "INFO")
        logger(f'Total size: {manifest["total_size_mb"]:.2f} MB', "INFO")
        logger(f'Total rows: {manifest["total_rows"]:,}', "INFO")

        return 0 if manifest["num_failed"] == 0 else 1

    except Exception as e:
        logger(f'Upload failed: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        spark.stop()
        logger('', "INFO")
        logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
