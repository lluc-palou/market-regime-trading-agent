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
        db_name: Database name
        input_collection: Source collection name
        output_collection: Destination collection name
        feature_names: List of all feature names
        final_transforms: Transform type per feature
        fitted_params: Fitted parameters per feature
    """
    from pyspark.sql.functions import col, expr
    from src.utils.logging import logger
    from src.feature_transformation.data_loader import get_all_hours, load_hour_batch

    # Build transformation map (feature index -> transform info)
    transform_map = {}
    for feat_name, transform_type in final_transforms.items():
        if transform_type != 'none' and feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            transform_map[feat_idx] = {
                'name': feat_name,
                'type': transform_type,
                'params': fitted_params.get(feat_name, {})
                }

    logger(f"Will transform {len(transform_map)} features", "INFO")
    logger("Using native Spark SQL (no UDFs) for better performance on large datasets", "INFO")

    # Show sample transformations
    logger("Sample transformations:", "INFO")
    for feat_name, transform_type in list(final_transforms.items())[:5]:
        logger(f"  {feat_name}: {transform_type}", "INFO")
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
                WHEN element_at(features, {feat_idx + 1}) = double('inf') THEN element_at(features, {feat_idx + 1})
                WHEN element_at(features, {feat_idx + 1}) = double('-inf') THEN element_at(features, {feat_idx + 1})
                ELSE {transform_expr}
            END
        """
        transform_expressions.append(safe_expr)

    # Combine all transformations into a single array
    features_array_expr = "array(" + ", ".join(transform_expressions) + ")"

    # Get all hours
    all_hours = get_all_hours(spark, db_name, input_collection)
    logger(f"Processing {len(all_hours)} hourly batches", "INFO")

    first_batch = True

    # Process each hour batch
    for hour_idx, start_hour in enumerate(all_hours):
        end_hour = start_hour + timedelta(hours=1)

        # Load hour batch with filters and ordering
        hour_df = load_hour_batch(
            spark,
            db_name,
            input_collection,
            start_hour,
            end_hour
        )

        hour_count = hour_df.count()
        if hour_count == 0:
            continue

        # Apply transformations using native Spark SQL
        transformed_df = hour_df.withColumn("features", expr(features_array_expr))

        # Drop _id to avoid conflicts
        if '_id' in transformed_df.columns:
            transformed_df = transformed_df.drop('_id')

        # Ensure temporal ordering before writing
        transformed_df = transformed_df.orderBy("timestamp")

        # Write batch to output collection with ordered writes
        mode = "overwrite" if first_batch else "append"

        (
            transformed_df
            .write
            .format("mongodb")
            .option("database", db_name)
            .option("collection", output_collection)
            .option("ordered", "true")
            .mode(mode)
            .save()
        )

        first_batch = False

        if (hour_idx + 1) % 10 == 0 or (hour_idx + 1) == len(all_hours):
            logger(
                f"  Batch {hour_idx + 1}/{len(all_hours)} "
                f"({start_hour.strftime('%Y-%m-%d %H:%M')}): "
                f"Processed {hour_count} docs",
                "INFO"
            )

    logger(f"Completed transformation of {len(all_hours)} batches", "INFO")


def fit_selected_transforms_on_full_data(
    spark,
    db_name: str,
    collection: str,
    feature_names: List[str],
    final_transforms: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Fit ONLY the selected transformations on 100% of training data.

    This is used in Stage 8 to re-fit the selected transforms (from Stage 7)
    on full training data instead of the 10% sample used in Stage 7.

    Args:
        spark: SparkSession
        db_name: Database name
        collection: Collection name (e.g., 'split_0_input')
        feature_names: List of all feature names
        final_transforms: Selected transform type per feature (from Stage 7)

    Returns:
        Dictionary of fitted parameters per feature
    """
    import numpy as np
    from src.utils.logging import logger
    from src.feature_transformation.data_loader import get_all_hours, load_hour_batch
    from src.feature_transformation.transforms import fit_transform_params

    logger(f"Fitting selected transforms on 100% of training data from {collection}", "INFO")

    # Build feature index map
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}

    # Initialize accumulators for training data
    train_data = {feat: [] for feat in final_transforms.keys() if feat in feature_names}

    # Get all hours
    all_hours = get_all_hours(spark, db_name, collection)
    logger(f"Processing {len(all_hours)} hours to collect training data", "INFO")

    # Collect training data from all hours
    for hour_idx, start_hour in enumerate(all_hours):
        end_hour = start_hour + timedelta(hours=1)

        hour_df = load_hour_batch(spark, db_name, collection, start_hour, end_hour)

        hour_count = hour_df.count()
        if hour_count == 0:
            continue

        # Collect to driver and filter training data
        rows = hour_df.select("features", "role").collect()

        for row in rows:
            # Only use training data
            if row['role'] != 'train':
                continue

            features_array = row['features']
            if features_array is None or len(features_array) != len(feature_names):
                continue

            # Collect values for each feature
            for feat_name in train_data.keys():
                feat_idx = feature_indices[feat_name]
                value = features_array[feat_idx]

                if value is not None:
                    try:
                        value = float(value)
                        if np.isfinite(value):
                            train_data[feat_name].append(value)
                    except (TypeError, ValueError):
                        pass

        if (hour_idx + 1) % 10 == 0:
            logger(f"  Collected data from {hour_idx + 1}/{len(all_hours)} hours", "INFO")

    # Fit transformations on collected training data
    logger("Fitting transformations on collected training data...", "INFO")
    fitted_params = {}

    for feat_name, transform_type in final_transforms.items():
        if feat_name not in train_data:
            logger(f"  Skipping {feat_name}: not in feature list", "WARNING")
            continue

        data = np.array(train_data[feat_name])

        if len(data) < 20:
            logger(f"  Skipping {feat_name}: insufficient data ({len(data)} samples)", "WARNING")
            continue

        # Fit the selected transform
        params = fit_transform_params(data, transform_type)
        if params is not None:
            fitted_params[feat_name] = params
            logger(f"  {feat_name}: fitted {transform_type} on {len(data):,} samples", "INFO")
        else:
            fitted_params[feat_name] = {}
            logger(f"  {feat_name}: {transform_type} needs no parameters", "INFO")

    logger(f"Fitted {len(fitted_params)} transformations on full training data", "INFO")

    return fitted_params
