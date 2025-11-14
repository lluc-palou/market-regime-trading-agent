"""
Export Splits to S3 (Stage 15)

Exports all split collections from MongoDB to S3 for checkpoint/portability.
Enables running different pipeline sections on different machines with different resources.

Workflow:
1. Discovers all split_X_input collections in MongoDB
2. Exports each split to S3 as Parquet (compressed, efficient)
3. Generates manifest.json with metadata (counts, checksums, schema)
4. Enables data versioning via run_id timestamps

Usage:
    python scripts/15_export_splits_to_s3.py
    python scripts/15_export_splits_to_s3.py --run-id custom_experiment_name
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
from pyspark.sql.functions import col, count as spark_count
from pymongo import MongoClient

from src.utils.logging import logger, log_section
from src.utils.spark import create_spark_session

# =================================================================================================
# AWS S3 Configuration
# =================================================================================================

S3_CONFIG = {
    "bucket": "your-drl-lob-bucket",           # S3 bucket name
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
    "app_name": "ExportSplitsToS3",
    "driver_memory": "8g",
    "jar_files_path": "file:///C:/Users/llucp/spark_jars/",
}

# =================================================================================================
# Helper Functions
# =================================================================================================

def discover_split_collections(mongo_uri: str, db_name: str) -> List[int]:
    """
    Discover all split_X_input collections in MongoDB.

    Returns:
        List of split IDs (sorted)
    """
    logger('Discovering split collections in MongoDB...', "INFO")

    client = MongoClient(mongo_uri)
    db = client[db_name]

    collections = db.list_collection_names()

    split_ids = []
    for coll in collections:
        if coll.startswith(MONGO_CONFIG["collection_prefix"]) and coll.endswith(MONGO_CONFIG["collection_suffix"]):
            # Extract split ID from collection name (e.g., "split_5_input" -> 5)
            try:
                split_id_str = coll.replace(MONGO_CONFIG["collection_prefix"], "").replace(MONGO_CONFIG["collection_suffix"], "")
                split_id = int(split_id_str)
                split_ids.append(split_id)
            except ValueError:
                logger(f'Warning: Could not parse split ID from collection {coll}', "WARNING")

    client.close()

    split_ids.sort()

    logger(f'Found {len(split_ids)} split collections: {split_ids}', "INFO")

    return split_ids


def get_split_statistics(spark: SparkSession, db_name: str, collection: str) -> Dict[str, Any]:
    """
    Get statistics for a split collection.

    Returns:
        Dictionary with document count, role distribution, schema info
    """
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .load()
    )

    # Get document count
    doc_count = df.count()

    # Get role distribution
    role_dist = {}
    if doc_count > 0 and "role" in df.columns:
        role_counts = df.groupBy("role").agg(spark_count("*").alias("count")).collect()
        role_dist = {row["role"]: row["count"] for row in role_counts}

    # Get schema info
    schema_fields = [field.name for field in df.schema.fields]

    # Get feature array length if present
    feature_length = None
    if doc_count > 0 and "features" in df.columns:
        first_row = df.select("features").first()
        if first_row and first_row["features"]:
            feature_length = len(first_row["features"])

    return {
        "num_documents": doc_count,
        "role_distribution": role_dist,
        "schema_fields": schema_fields,
        "feature_length": feature_length,
    }


def export_split_to_s3(
    spark: SparkSession,
    db_name: str,
    collection: str,
    s3_path: str,
    split_id: int
) -> Dict[str, Any]:
    """
    Export a single split collection to S3 as Parquet.

    Returns:
        Dictionary with export statistics
    """
    logger(f'Exporting split {split_id} to S3...', "INFO")
    logger(f'  Source: {db_name}.{collection}', "INFO")
    logger(f'  Destination: {s3_path}', "INFO")

    # Read from MongoDB
    df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .load()
    )

    # Get statistics before export
    stats = get_split_statistics(spark, db_name, collection)

    if stats["num_documents"] == 0:
        logger(f'  Split {split_id} is empty, skipping export', "WARNING")
        return None

    # Convert ObjectId to string for Parquet compatibility
    if "_id" in df.columns:
        df = df.withColumn("_id", col("_id").cast("string"))

    # Write to S3 as Parquet with compression
    df.write.mode("overwrite").parquet(s3_path)

    logger(f'  Exported {stats["num_documents"]:,} documents', "INFO")
    logger(f'  Features: {stats["feature_length"]} dimensions', "INFO")
    logger(f'  Role distribution: {stats["role_distribution"]}', "INFO")

    return {
        "split_id": split_id,
        "collection": collection,
        "s3_path": s3_path,
        **stats
    }


def calculate_s3_checksum(spark: SparkSession, s3_path: str) -> str:
    """
    Calculate checksum for S3 Parquet files.

    This is a simple implementation - in production you might want to use S3 ETags.
    """
    try:
        # Read back from S3 and calculate basic checksum on document count + schema
        df = spark.read.parquet(s3_path)
        count = df.count()
        schema_str = str(df.schema)
        checksum_input = f"{count}:{schema_str}"
        checksum = hashlib.md5(checksum_input.encode()).hexdigest()
        return f"md5:{checksum}"
    except Exception as e:
        logger(f'Warning: Could not calculate checksum: {e}', "WARNING")
        return "unknown"


def generate_manifest(
    run_id: str,
    split_stats: List[Dict[str, Any]],
    pipeline_stage: int = 14
) -> Dict[str, Any]:
    """
    Generate manifest file with metadata about the export.
    """
    return {
        "run_id": run_id,
        "export_timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_stage": pipeline_stage,
        "database": MONGO_CONFIG["db_name"],
        "num_splits": len(split_stats),
        "format": "parquet",
        "compression": "snappy",
        "s3_config": {
            "bucket": S3_CONFIG["bucket"],
            "prefix": S3_CONFIG["prefix"],
            "region": S3_CONFIG["region"],
        },
        "splits": split_stats,
        "total_documents": sum(s["num_documents"] for s in split_stats if s),
    }


def upload_manifest_to_s3(spark: SparkSession, manifest: Dict[str, Any], s3_manifest_path: str):
    """
    Upload manifest JSON to S3.
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


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""

    # Parse command line arguments for custom run_id
    import argparse
    parser = argparse.ArgumentParser(description='Export splits to S3')
    parser.add_argument('--run-id', type=str, default=None, help='Custom run ID (default: timestamp)')
    args = parser.parse_args()

    # Generate run ID (timestamp or custom)
    run_id = args.run_id if args.run_id else datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    log_section('EXPORT SPLITS TO S3 (STAGE 15)')
    logger('', "INFO")

    logger('Configuration:', "INFO")
    logger(f'  MongoDB: {MONGO_CONFIG["uri"]} / {MONGO_CONFIG["db_name"]}', "INFO")
    logger(f'  S3 Bucket: {S3_CONFIG["bucket"]}', "INFO")
    logger(f'  S3 Prefix: {S3_CONFIG["prefix"]}', "INFO")
    logger(f'  S3 Region: {S3_CONFIG["region"]}', "INFO")
    logger(f'  Run ID: {run_id}', "INFO")
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
        # Step 1: Discover Splits
        # =====================================================================

        log_section('STEP 1: DISCOVER SPLITS')
        split_ids = discover_split_collections(MONGO_CONFIG["uri"], MONGO_CONFIG["db_name"])

        if not split_ids:
            logger('No split collections found! Exiting.', "ERROR")
            return 1

        logger('', "INFO")

        # =====================================================================
        # Step 2: Export Splits to S3
        # =====================================================================

        log_section('STEP 2: EXPORT SPLITS TO S3')

        split_stats = []

        for i, split_id in enumerate(split_ids):
            logger(f'[{i+1}/{len(split_ids)}] Processing split {split_id}...', "INFO")

            collection_name = f"{MONGO_CONFIG['collection_prefix']}{split_id}{MONGO_CONFIG['collection_suffix']}"
            s3_path = f"s3a://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/{run_id}/split_{split_id}.parquet"

            stats = export_split_to_s3(
                spark,
                MONGO_CONFIG["db_name"],
                collection_name,
                s3_path,
                split_id
            )

            if stats:
                # Calculate checksum
                stats["checksum"] = calculate_s3_checksum(spark, s3_path)
                split_stats.append(stats)

            logger('', "INFO")

        # =====================================================================
        # Step 3: Generate and Upload Manifest
        # =====================================================================

        log_section('STEP 3: GENERATE MANIFEST')

        manifest = generate_manifest(run_id, split_stats)

        logger('Manifest summary:', "INFO")
        logger(f'  Run ID: {manifest["run_id"]}', "INFO")
        logger(f'  Timestamp: {manifest["export_timestamp"]}', "INFO")
        logger(f'  Splits: {manifest["num_splits"]}', "INFO")
        logger(f'  Total documents: {manifest["total_documents"]:,}', "INFO")
        logger('', "INFO")

        # Upload manifest to S3
        s3_manifest_path = f"s3a://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/{run_id}/manifest.json"
        upload_manifest_to_s3(spark, manifest, s3_manifest_path)

        # Also save manifest locally for reference
        local_manifest_path = os.path.join(REPO_ROOT, "artifacts", "s3_exports", f"manifest_{run_id}.json")
        os.makedirs(os.path.dirname(local_manifest_path), exist_ok=True)
        with open(local_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger(f'Manifest also saved locally: {local_manifest_path}', "INFO")

        logger('', "INFO")

        # =====================================================================
        # Completion
        # =====================================================================

        log_section('EXPORT COMPLETE')
        logger(f'Successfully exported {len(split_stats)} splits to S3', "INFO")
        logger(f'S3 Location: s3://{S3_CONFIG["bucket"]}/{S3_CONFIG["prefix"]}/{run_id}/', "INFO")
        logger(f'Run ID: {run_id}', "INFO")
        logger('', "INFO")
        logger('Next steps:', "INFO")
        logger(f'  1. Run on different machine: python scripts/16_import_splits_from_s3.py --run-id {run_id}', "INFO")
        logger('  2. Continue pipeline from Stage 17+ (modeling/training)', "INFO")

        return 0

    except Exception as e:
        logger(f'Export failed: {str(e)}', "ERROR")
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
