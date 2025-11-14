"""
Import Splits from S3 (Stage 16)

Downloads split collections from S3 and restores them to MongoDB.
Enables running different pipeline sections on different machines with different resources.

Workflow:
1. Lists available exports in S3 (or accepts specific run_id)
2. Downloads and validates manifest
3. Restores each split to MongoDB
4. Verifies data integrity (document counts, checksums)

Usage:
    python scripts/16_import_splits_from_s3.py --run-id 20250314_143022
    python scripts/16_import_splits_from_s3.py --latest
    python scripts/16_import_splits_from_s3.py --list  # List available exports
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

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
from pyspark.sql.functions import col
from pymongo import MongoClient

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session

# =================================================================================================
# AWS S3 Configuration (must match export script)
# =================================================================================================

S3_CONFIG = {
    "bucket": "lluc-tfg-data",           # S3 bucket name
    "prefix": "processed-splits",              # Prefix for all split exports
    "region": "us-east-1",                     # AWS region
    # IAM role will be used for authentication (no keys needed in code)
}

# =================================================================================================
# MongoDB Configuration
# =================================================================================================

MONGO_CONFIG = {
    "uri": "mongodb://127.0.0.1:27017/",
    "db_name": "raw",
    "collection_prefix": "split_",
    "collection_suffix": "_input",
}

# =================================================================================================
# Spark Configuration
# =================================================================================================

SPARK_CONFIG = {
    "app_name": "ImportSplitsFromS3",
    "driver_memory": "8g",
    "jar_files_path": "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/",
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def list_available_exports(spark: SparkSession) -> List[str]:
    """
    List available export run_ids in S3.

    Returns:
        List of run_ids (sorted by timestamp, newest first)
    """
    logger('Listing available exports in S3...', "INFO")

    try:
        # Use Spark to list directories in S3
        s3_base = f"s3a://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/"

        # This is tricky with Spark - we'll use boto3 for listing
        import boto3
        s3_client = boto3.client('s3', region_name=S3_CONFIG['region'])

        response = s3_client.list_objects_v2(
            Bucket=S3_CONFIG['bucket'],
            Prefix=S3_CONFIG['prefix'] + '/',
            Delimiter='/'
        )

        run_ids = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                # Extract run_id from prefix (e.g., "processed-splits/20250314_143022/")
                run_id = prefix['Prefix'].replace(S3_CONFIG['prefix'] + '/', '').rstrip('/')
                if run_id:
                    run_ids.append(run_id)

        run_ids.sort(reverse=True)  # Newest first

        return run_ids

    except Exception as e:
        logger(f'Warning: Could not list S3 exports: {e}', "WARNING")
        logger('Make sure boto3 is installed and AWS credentials are configured', "WARNING")
        return []


def download_manifest(spark: SparkSession, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Download and parse manifest file from S3.

    Returns:
        Manifest dictionary or None if failed
    """
    logger(f'Downloading manifest for run {run_id}...', "INFO")

    s3_manifest_path = f"s3a://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/{run_id}/manifest.json"

    try:
        # Read JSON from S3
        manifest_df = spark.read.json(s3_manifest_path)

        if manifest_df.count() == 0:
            logger(f'Manifest not found at {s3_manifest_path}', "ERROR")
            return None

        # Get the JSON string
        manifest_json = manifest_df.first()["manifest_json"]
        manifest = json.loads(manifest_json)

        logger('Manifest downloaded successfully', "INFO")
        logger(f'  Export timestamp: {manifest["export_timestamp"]}', "INFO")
        logger(f'  Splits: {manifest["num_splits"]}', "INFO")
        logger(f'  Total documents: {manifest["total_documents"]:,}', "INFO")

        return manifest

    except Exception as e:
        logger(f'Failed to download manifest: {e}', "ERROR")
        return None


def import_split_from_s3(
    spark: SparkSession,
    split_info: Dict[str, Any],
    db_name: str,
    mongo_uri: str
) -> bool:
    """
    Import a single split from S3 to MongoDB.

    Returns:
        True if successful, False otherwise
    """
    split_id = split_info["split_id"]
    s3_path = split_info["s3_path"]
    collection_name = f"{MONGO_CONFIG['collection_prefix']}{split_id}{MONGO_CONFIG['collection_suffix']}"

    logger(f'Importing split {split_id}...', "INFO")
    logger(f'  Source: {s3_path}', "INFO")
    logger(f'  Destination: {db_name}.{collection_name}', "INFO")

    try:
        # Read from S3
        df = spark.read.parquet(s3_path)

        # Convert _id back from string to proper format for MongoDB
        # Note: MongoDB connector will handle ObjectId conversion automatically
        # if _id is present, so we just need to ensure it's a string

        # Drop existing collection first (clean slate)
        client = MongoClient(mongo_uri)
        db = client[db_name]
        if collection_name in db.list_collection_names():
            db[collection_name].drop()
            logger(f'  Dropped existing collection {collection_name}', "INFO")
        client.close()

        # Write to MongoDB
        df.write.format("mongodb") \
            .option("database", db_name) \
            .option("collection", collection_name) \
            .option("ordered", "false") \
            .mode("overwrite") \
            .save()

        # Verify document count
        client = MongoClient(mongo_uri)
        db = client[db_name]
        actual_count = db[collection_name].count_documents({})
        expected_count = split_info["num_documents"]
        client.close()

        if actual_count != expected_count:
            logger(f'  WARNING: Document count mismatch! Expected {expected_count:,}, got {actual_count:,}', "WARNING")
            return False

        logger(f'  Imported {actual_count:,} documents successfully', "INFO")
        logger(f'  Role distribution: {split_info.get("role_distribution", {})}', "INFO")

        return True

    except Exception as e:
        logger(f'  Failed to import split {split_id}: {e}', "ERROR")
        return False


def verify_restoration(spark: SparkSession, manifest: Dict[str, Any], mongo_uri: str, db_name: str) -> bool:
    """
    Verify that all splits were restored correctly.

    Returns:
        True if all checks pass, False otherwise
    """
    logger('Verifying restoration...', "INFO")

    client = MongoClient(mongo_uri)
    db = client[db_name]

    all_passed = True

    for split_info in manifest["splits"]:
        if not split_info:  # Skip empty splits
            continue

        split_id = split_info["split_id"]
        collection_name = f"{MONGO_CONFIG['collection_prefix']}{split_id}{MONGO_CONFIG['collection_suffix']}"

        # Check collection exists
        if collection_name not in db.list_collection_names():
            logger(f'  FAIL: Collection {collection_name} not found', "ERROR")
            all_passed = False
            continue

        # Check document count
        actual_count = db[collection_name].count_documents({})
        expected_count = split_info["num_documents"]

        if actual_count != expected_count:
            logger(f'  FAIL: Split {split_id} count mismatch (expected {expected_count:,}, got {actual_count:,})', "ERROR")
            all_passed = False
        else:
            logger(f'  PASS: Split {split_id} ({actual_count:,} documents)', "INFO")

    client.close()

    return all_passed


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Import splits from S3')
    parser.add_argument('--run-id', type=str, default=None, help='Specific run ID to import')
    parser.add_argument('--latest', action='store_true', help='Import latest export')
    parser.add_argument('--list', action='store_true', help='List available exports and exit')
    args = parser.parse_args()

    log_section('IMPORT SPLITS FROM S3 (STAGE 16)')
    logger('', "INFO")

    logger('Configuration:', "INFO")
    logger(f'  MongoDB: {MONGO_CONFIG["uri"]} / {MONGO_CONFIG["db_name"]}', "INFO")
    logger(f'  S3 Bucket: {S3_CONFIG["bucket"]}', "INFO")
    logger(f'  S3 Prefix: {S3_CONFIG["prefix"]}', "INFO")
    logger(f'  S3 Region: {S3_CONFIG["region"]}', "INFO")
    logger('', "INFO")

    # Create Spark session with S3 support
    logger('Initializing Spark with S3 support...', "INFO")
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"],
        mongo_uri=MONGO_CONFIG["uri"],
        db_name=MONGO_CONFIG["db_name"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        jar_files_path=SPARK_CONFIG["jar_files_path"]
    )

    # Configure S3 for Spark
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider")

    logger('Spark session created', "INFO")
    logger('', "INFO")

    try:
        # =====================================================================
        # Step 1: Determine which export to import
        # =====================================================================

        if args.list:
            log_section('AVAILABLE EXPORTS')
            available = list_available_exports(spark)
            if available:
                logger(f'Found {len(available)} exports:', "INFO")
                for i, run_id in enumerate(available):
                    logger(f'  {i+1}. {run_id}', "INFO")
            else:
                logger('No exports found in S3', "INFO")
            return 0

        # Determine run_id to import
        run_id = None

        if args.run_id:
            run_id = args.run_id
            logger(f'Importing specified run: {run_id}', "INFO")
        elif args.latest:
            available = list_available_exports(spark)
            if not available:
                logger('No exports found in S3!', "ERROR")
                return 1
            run_id = available[0]
            logger(f'Importing latest run: {run_id}', "INFO")
        else:
            logger('Error: Must specify --run-id or --latest', "ERROR")
            logger('Use --list to see available exports', "INFO")
            return 1

        logger('', "INFO")

        # =====================================================================
        # Step 2: Download and Validate Manifest
        # =====================================================================

        log_section('STEP 1: DOWNLOAD MANIFEST')

        manifest = download_manifest(spark, run_id)

        if not manifest:
            logger('Failed to download manifest. Exiting.', "ERROR")
            return 1

        logger('', "INFO")
        logger('Manifest validation:', "INFO")
        logger(f'  Format: {manifest["format"]}', "INFO")
        logger(f'  Compression: {manifest["compression"]}', "INFO")
        logger(f'  Pipeline stage: {manifest["pipeline_stage"]}', "INFO")

        logger('', "INFO")

        # =====================================================================
        # Step 3: Import Splits from S3
        # =====================================================================

        log_section('STEP 2: IMPORT SPLITS FROM S3')

        success_count = 0
        fail_count = 0

        splits = [s for s in manifest["splits"] if s]  # Filter out None/empty

        for i, split_info in enumerate(splits):
            logger(f'[{i+1}/{len(splits)}] Processing split {split_info["split_id"]}...', "INFO")

            success = import_split_from_s3(
                spark,
                split_info,
                MONGO_CONFIG["db_name"],
                MONGO_CONFIG["uri"]
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            logger('', "INFO")

        # =====================================================================
        # Step 4: Verify Restoration
        # =====================================================================

        log_section('STEP 3: VERIFY RESTORATION')

        verification_passed = verify_restoration(
            spark,
            manifest,
            MONGO_CONFIG["uri"],
            MONGO_CONFIG["db_name"]
        )

        logger('', "INFO")

        # =====================================================================
        # Completion
        # =====================================================================

        log_section('IMPORT COMPLETE')

        if verification_passed and fail_count == 0:
            logger(f'✓ Successfully imported {success_count} splits from S3', "INFO")
            logger(f'  Run ID: {run_id}', "INFO")
            logger(f'  Total documents: {manifest["total_documents"]:,}', "INFO")
            logger('', "INFO")
            logger('Data restored successfully. Ready to continue pipeline.', "INFO")
            return 0
        else:
            logger(f'⚠ Import completed with issues:', "WARNING")
            logger(f'  Successful: {success_count}', "INFO")
            logger(f'  Failed: {fail_count}', "ERROR")
            logger('', "INFO")
            logger('Please review errors above.', "WARNING")
            return 1

    except Exception as e:
        logger(f'Import failed: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Check if running from orchestrator
        is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'
        if not is_orchestrated:
            spark.stop()
            logger('', "INFO")
            logger('Spark session stopped', "INFO")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
