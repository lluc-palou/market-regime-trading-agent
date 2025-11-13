"""
Vectorized Batch Transformation for Stage 8

Uses pandas UDFs for efficient distributed transformation.
Processes data in batches without collecting to driver.
"""

import math
import numpy as np
from typing import Dict, List
from datetime import timedelta


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
    Apply transformations using pandas UDFs for efficient distributed processing.

    Args:
        spark: SparkSession
        db_name: MongoDB database name
        input_collection: Input collection name
        output_collection: Output collection name
        feature_names: List of feature names
        final_transforms: Transform type per feature
        fitted_params: Fitted parameters per feature
    """
    # Import here to avoid module-level imports that leak into UDF serialization
    import math
    import numpy as np
    import pandas as pd
    from pyspark.sql.functions import pandas_udf, col
    from pyspark.sql.types import ArrayType, DoubleType
    from src.utils.logging import logger
    from src.feature_transformation.data_loader import get_all_hours, load_hour_batch

    # Build transformation map (feature_idx -> transform info)
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

    # Create pandas UDF for vectorized transformation
    # Must be self-contained (no external function calls in workers)
    # Import standard libraries inside UDF to ensure they're available in workers
    @pandas_udf(ArrayType(DoubleType()))
    def transform_features_udf(features_series: pd.Series) -> pd.Series:
        """
        Vectorized transformation using pandas UDF.
        Processes batches of rows efficiently.
        Self-contained - all logic embedded to work in worker processes.
        """
        def apply_transform(value: float, transform_type: str, params: dict) -> float:
            """Apply single transformation (embedded in UDF for worker compatibility)."""
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
                result = float(np.arcsinh(value))

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

            if result is not None:
                return float(result)
            return value

        def transform_row(features_array):
            if features_array is None:
                return features_array

            features_list = list(features_array)

            for feat_idx, info in transform_map.items():
                if feat_idx >= len(features_list):
                    continue

                original_value = features_list[feat_idx]
                if original_value is None or pd.isna(original_value):
                    continue

                try:
                    transformed_value = apply_transform(
                        float(original_value),
                        info['type'],
                        info['params']
                    )

                    if transformed_value is not None:
                        if not (np.isnan(transformed_value) or np.isinf(transformed_value)):
                            features_list[feat_idx] = float(transformed_value)

                except Exception:
                    pass

            return features_list

        return features_series.apply(transform_row)

    # Get all hourly windows
    hours = get_all_hours(spark, db_name, input_collection)
    if not hours:
        logger("No time windows found, aborting", "ERROR")
        return

    logger(f"Processing {len(hours)} hours of data", "INFO")

    # Process hour by hour
    total_processed = 0
    first_batch = True

    for hour_idx, start_hour in enumerate(hours):
        end_hour = start_hour + timedelta(hours=1)

        # Load hour batch
        hour_df = load_hour_batch(spark, db_name, input_collection, start_hour, end_hour)
        hour_count = hour_df.count()

        if hour_count == 0:
            continue

        logger(f"Hour {hour_idx + 1}/{len(hours)}: {start_hour.strftime('%Y-%m-%d %H:00')} - {hour_count:,} samples", "INFO")

        # Apply transformations using pandas UDF (distributed, vectorized)
        transformed_df = hour_df.withColumn('features', transform_features_udf(col('features')))

        # Drop _id if present (MongoDB internal field)
        if '_id' in transformed_df.columns:
            transformed_df = transformed_df.drop('_id')

        # Sort by timestamp to ensure ordering
        transformed_df = transformed_df.orderBy("timestamp")

        # Write with ordered writes to preserve time order
        write_mode = "overwrite" if first_batch else "append"

        (transformed_df.write.format("mongodb")
         .option("database", db_name)
         .option("collection", output_collection)
         .option("ordered", "true")
         .mode(write_mode)
         .save())

        total_processed += hour_count
        first_batch = False

        logger(f"  Transformed and wrote {hour_count:,} samples in time order", "INFO")

    logger("=" * 80, "INFO")
    logger(f"Successfully processed {total_processed:,} documents", "INFO")
    logger(f"Written to {output_collection} in chronological order", "INFO")
    logger("=" * 80, "INFO")


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