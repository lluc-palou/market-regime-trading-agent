"""
Direct Batch Transformation for Stage 8

Uses Stage 7's proven data loading functions for hour-by-hour processing.
"""

from typing import Dict, List
from datetime import timedelta
import math
import numpy as np

from src.utils.logging import logger
from src.utils.database import write_to_mongodb
from src.feature_transformation.data_loader import get_all_hours, load_hour_batch


def apply_transformations_direct(
    spark,
    db_name: str,
    input_collection: str,
    output_collection: str,
    feature_names: List[str],
    final_transforms: Dict[str, str],
    fitted_params: Dict[str, Dict]
) -> None:
    """
    Apply transformations hour-by-hour using Stage 7's data loading approach.
    
    Args:
        spark: SparkSession
        db_name: MongoDB database name
        input_collection: Input collection name
        output_collection: Output collection name
        feature_names: List of feature names
        final_transforms: Transform type per feature
        fitted_params: Fitted parameters per feature
    """
    
    # Build transformation map
    transform_map = {}
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_name in final_transforms:
            transform_type = final_transforms[feat_name]
            if transform_type not in ('none', 'identity'):
                params = fitted_params.get(feat_name, {}).get(transform_type, {})
                transform_map[feat_idx] = {
                    'type': transform_type,
                    'params': params,
                    'name': feat_name
                }
    
    logger(f"Will transform {len(transform_map)} features", "INFO")
    
    # Show sample transformations
    logger("Sample transformations:", "INFO")
    for idx, (feat_idx, info) in enumerate(list(transform_map.items())[:5]):
        logger(f"  Feature {feat_idx} ({info['name']}): {info['type']} with params {info['params']}", "INFO")
    if len(transform_map) > 5:
        logger(f"  ... and {len(transform_map) - 5} more", "INFO")
    
    # Get all hourly windows using Stage 7's function
    hours = get_all_hours(spark, db_name, input_collection)
    if not hours:
        logger("No time windows found, aborting", "ERROR")
        return
    
    logger(f"Processing {len(hours)} hours of data", "INFO")
    
    # Process hour by hour
    total_processed = 0
    sample_before = None
    sample_after = None
    first_batch = True
    
    for hour_idx, start_hour in enumerate(hours):
        end_hour = start_hour + timedelta(hours=1)
        
        # Load hour batch using Stage 7's function
        hour_df = load_hour_batch(spark, db_name, input_collection, start_hour, end_hour)
        hour_count = hour_df.count()
        
        if hour_count == 0:
            continue
        
        logger(f"Hour {hour_idx + 1}/{len(hours)}: {start_hour.strftime('%Y-%m-%d %H:00')} - {hour_count:,} samples", "INFO")
        
        # Collect hour data (already sorted by timestamp from pipeline)
        rows = hour_df.collect()
        transformed_rows = []
        
        for row in rows:
            row_dict = row.asDict()
            
            # Convert timestamp to UTC naive (Spark/MongoDB may add local timezone)
            ts = row_dict.get('timestamp')
            if ts and hasattr(ts, 'tzinfo'):
                if ts.tzinfo is not None:
                    # Has timezone - convert to UTC and strip tzinfo
                    import pytz
                    if ts.tzinfo != pytz.UTC:
                        ts = ts.astimezone(pytz.UTC)
                    ts = ts.replace(tzinfo=None)
                    row_dict['timestamp'] = ts
            
            features = row_dict.get('features')
            
            if features is None:
                transformed_rows.append(row_dict)
                continue
            
            # Save first sample for verification
            if sample_before is None:
                sample_before = list(features[:5])
                # Format timestamp as UTC string (now guaranteed UTC naive)
                if ts:
                    ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' if hasattr(ts, 'strftime') else str(ts)
                    logger(f"  First sample timestamp: {ts_str}", "INFO")
            
            # Transform features
            features_list = list(features)
            
            for feat_idx, info in transform_map.items():
                if feat_idx >= len(features_list):
                    continue
                
                original_value = features_list[feat_idx]
                if original_value is None:
                    continue
                
                try:
                    transformed_value = apply_single_transform(
                        original_value, info['type'], info['params']
                    )
                    
                    if transformed_value is not None:
                        if not (math.isnan(transformed_value) or math.isinf(transformed_value)):
                            features_list[feat_idx] = transformed_value
                
                except Exception:
                    pass
            
            # Save first transformed sample
            if sample_after is None:
                sample_after = features_list[:5]
            
            # Convert to native Python floats
            row_dict['features'] = [float(v) if v is not None else None for v in features_list]
            transformed_rows.append(row_dict)
        
        # Verify we got the data
        if not transformed_rows:
            logger(f"  WARNING: No data transformed for this hour!", "WARNING")
            continue
        
        # Convert back to DataFrame - use schema from original to preserve types
        # This prevents Spark from inferring timestamps with timezone
        transformed_df = spark.createDataFrame(transformed_rows, schema=hour_df.schema)
        
        # Verify timestamp order is preserved
        first_ts = transformed_rows[0].get('timestamp')
        last_ts = transformed_rows[-1].get('timestamp')
        
        # Format as UTC strings (strip any timezone info)
        def format_utc(ts):
            if ts is None:
                return "None"
            # Strip timezone if present
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            # Format as UTC
            if hasattr(ts, 'strftime'):
                return ts.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            return str(ts)
        
        first_ts_str = format_utc(first_ts)
        last_ts_str = format_utc(last_ts)
        
        logger(f"  Time range: {first_ts_str} to {last_ts_str}", "INFO")
        
        # Drop _id if present (MongoDB internal field)
        if '_id' in transformed_df.columns:
            transformed_df = transformed_df.drop('_id')
        
        # Sort by timestamp to ensure ordering (like Stage 6)
        transformed_df = transformed_df.orderBy("timestamp")
        
        # Write with ordered writes to preserve time order (like Stage 6)
        write_mode = "overwrite" if first_batch else "append"
        
        (transformed_df.write.format("mongodb")
         .option("database", db_name)
         .option("collection", output_collection)
         .option("ordered", "true")  # Preserve insertion order
         .mode(write_mode)
         .save())
        
        total_processed += len(transformed_rows)
        first_batch = False
        
        logger(f"  Transformed and wrote {len(transformed_rows):,} samples in time order", "INFO")
    
    # Verification
    logger("=" * 80, "INFO")
    logger("TRANSFORMATION VERIFICATION", "INFO")
    logger("=" * 80, "INFO")
    logger(f"BEFORE: {sample_before}", "INFO")
    logger(f"AFTER:  {sample_after}", "INFO")
    
    changes = sum(1 for b, a in zip(sample_before or [], sample_after or []) if b != a)
    if changes > 0:
        logger(f"✓ Detected {changes} changes in first 5 features", "INFO")
    else:
        logger("WARNING: No changes detected in sample!", "WARNING")
    logger("=" * 80, "INFO")
    
    logger(f"Successfully processed {total_processed:,} documents", "INFO")
    logger(f"Written to {output_collection} in chronological order", "INFO")


def apply_single_transform(value: float, transform_type: str, params: Dict) -> float:
    """
    Apply a single transformation to a value.
    
    Args:
        value: Original value
        transform_type: Type of transformation
        params: Parameters for transformation
        
    Returns:
        Transformed value (native Python float)
    """
    result = None
    
    if transform_type == 'log':
        offset = params.get('offset', 0.0)
        if (value + offset) > 0:
            result = math.log(value + offset)
    
    elif transform_type == 'log1p':
        offset = params.get('offset', 0.0)
        if (value + offset) >= 0:
            result = math.log(value + offset + 1.0)
    
    elif transform_type == 'sqrt':
        offset = params.get('offset', 0.0)
        if (value + offset) >= 0:
            result = math.sqrt(value + offset)
    
    elif transform_type == 'arcsinh':
        result = float(np.arcsinh(value))  # Convert numpy to Python float
    
    elif transform_type == 'box_cox':
        lambda_param = params.get('lambda', 1.0)
        offset = params.get('offset', 0.0)
        
        if (value + offset) > 0:
            if abs(lambda_param) < 1e-10:
                result = math.log(value + offset)
            else:
                result = (math.pow(value + offset, lambda_param) - 1.0) / lambda_param
    
    elif transform_type == 'yeo_johnson':
        lambda_param = params.get('lambda', 1.0)
        
        if value >= 0:
            if abs(lambda_param) < 1e-10:
                result = math.log(value + 1.0)
            else:
                result = (math.pow(value + 1.0, lambda_param) - 1.0) / lambda_param
        else:
            if abs(lambda_param - 2.0) < 1e-10:
                result = -math.log(-value + 1.0)
            else:
                result = -(math.pow(-value + 1.0, 2.0 - lambda_param) - 1.0) / (2.0 - lambda_param)
    
    # Ensure result is native Python float
    if result is not None:
        return float(result)
    
    return value