"""
Vectorized Batch Transformation for Stage 8

Uses native Spark SQL for efficient distributed transformation.
Processes data in batches without collecting to driver.
"""

from typing import Dict, List
from datetime import timedelta


def _build_transform_sql(feat_idx: int, transform_type: str, params: Dict) -> str:
    """
    Build SQL expression for a single feature transformation.

    Args:
        feat_idx: Feature index
        transform_type: Type of transformation
        params: Transformation parameters

    Returns:
        SQL expression string for the transformation
    """
    value_expr = f"element_at(features, {feat_idx + 1})"

    if transform_type in ('none', 'identity'):
        return value_expr

    elif transform_type == 'log':
        offset = params.get('offset', 0.0)
        return f"""
            CASE
                WHEN ({value_expr} + {offset}) > 0
                THEN ln({value_expr} + {offset})
                ELSE {value_expr}
            END
        """

    elif transform_type == 'log1p':
        offset = params.get('offset', 0.0)
        return f"""
            CASE
                WHEN ({value_expr} + {offset}) >= 0
                THEN ln({value_expr} + {offset} + 1.0)
                ELSE {value_expr}
            END
        """

    elif transform_type == 'sqrt':
        offset = params.get('offset', 0.0)
        return f"""
            CASE
                WHEN ({value_expr} + {offset}) >= 0
                THEN sqrt({value_expr} + {offset})
                ELSE {value_expr}
            END
        """

    elif transform_type == 'arcsinh':
        # arcsinh(x) = ln(x + sqrt(x^2 + 1))
        return f"ln({value_expr} + sqrt(power({value_expr}, 2) + 1))"

    elif transform_type == 'box_cox':
        lambda_param = params.get('lambda', 1.0)
        offset = params.get('offset', 0.0)

        if abs(lambda_param) < 1e-10:
            # Lambda ~= 0: use log
            return f"""
                CASE
                    WHEN ({value_expr} + {offset}) > 0
                    THEN ln({value_expr} + {offset})
                    ELSE {value_expr}
                END
            """
        else:
            # Lambda != 0: (x^lambda - 1) / lambda
            return f"""
                CASE
                    WHEN ({value_expr} + {offset}) > 0
                    THEN (power({value_expr} + {offset}, {lambda_param}) - 1.0) / {lambda_param}
                    ELSE {value_expr}
                END
            """

    elif transform_type == 'yeo_johnson':
        lambda_param = params.get('lambda', 1.0)

        # Yeo-Johnson has two branches for positive and negative values
        if abs(lambda_param) < 1e-10:
            # Lambda ~= 0, value >= 0
            positive_branch = f"ln({value_expr} + 1.0)"
        else:
            # Lambda != 0, value >= 0
            positive_branch = f"(power({value_expr} + 1.0, {lambda_param}) - 1.0) / {lambda_param}"

        if abs(lambda_param - 2.0) < 1e-10:
            # Lambda ~= 2, value < 0
            negative_branch = f"-ln(-{value_expr} + 1.0)"
        else:
            # Lambda != 2, value < 0
            negative_branch = f"-(power(-{value_expr} + 1.0, {2.0 - lambda_param}) - 1.0) / {2.0 - lambda_param}"

        return f"""
            CASE
                WHEN {value_expr} >= 0 THEN {positive_branch}
                ELSE {negative_branch}
            END
        """

    else:
        # Unknown transform type, return original value
        return value_expr


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
    Apply transformations using native Spark SQL for efficient distributed processing.

    Args:
        spark: SparkSession
        db_name: MongoDB database name
        input_collection: Input collection name
        output_collection: Output collection name
        feature_names: List of feature names
        final_transforms: Transform type per feature
        fitted_params: Fitted parameters per feature
    """
    from pyspark.sql.functions import col, expr
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
    logger("Using native Spark SQL (no UDFs) for better performance on large datasets", "INFO")

    # Show sample transformations
    logger("Sample transformations:", "INFO")
    for idx, (feat_idx, info) in enumerate(list(transform_map.items())[:5]):
        logger(f"  Feature {feat_idx} ({info['name']}): {info['type']} with params {info['params']}", "INFO")
    if len(transform_map) > 5:
        logger(f"  ... and {len(transform_map) - 5} more", "INFO")

    # Build native Spark SQL transformation expression
    # Use transform() with index to selectively apply transformations
    if len(transform_map) == 0:
        logger("No transformations to apply, skipping", "WARNING")
        return

    # Build array of transformed features using SQL CASE for each index
    transform_expressions = []
    for feat_idx in range(len(feature_names)):
        if feat_idx in transform_map:
            info = transform_map[feat_idx]
            transform_expr = _build_transform_sql(feat_idx, info['type'], info['params'])
        else:
            # No transformation for this feature, keep original
            transform_expr = f"element_at(features, {feat_idx + 1})"

        # Wrap with null/nan/inf protection
        safe_expr = f"""
            CASE
                WHEN element_at(features, {feat_idx + 1}) IS NULL THEN NULL
                WHEN isnan(element_at(features, {feat_idx + 1})) THEN element_at(features, {feat_idx + 1})
                WHEN isinf(element_at(features, {feat_idx + 1})) THEN element_at(features, {feat_idx + 1})
                ELSE {transform_expr}
            END
        """
        transform_expressions.append(safe_expr)

    # Combine all expressions into an array
    full_transform_expr = "array(" + ", ".join(transform_expressions) + ")"

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

        # Apply transformations using native Spark SQL (no UDFs)
        transformed_df = hour_df.withColumn('features', expr(full_transform_expr))

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