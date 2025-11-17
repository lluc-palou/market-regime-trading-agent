"""
Feature Transformation Selection Module

Selects optimal normalization transformations for LOB features using CPCV.
Uses true two-pass processing without full data accumulation.
"""

from .transforms import (
    fit_transform_params,
    apply_transform,
    compute_pearson_p_df,
    TRANSFORM_TYPES
)
from .data_loader import (
    get_all_hours,
    load_hour_batch,
    identify_feature_names,
    identify_feature_names_from_collection
)
from .processor import FeatureTransformProcessor
from .aggregator import aggregate_across_splits, select_final_transforms
from .transformation_application import apply_transformations_direct

__all__ = [
    'fit_transform_params',
    'apply_transform',
    'compute_pearson_p_df',
    'TRANSFORM_TYPES',
    'get_all_hours',
    'load_hour_batch',
    'identify_feature_names',
    'identify_feature_names_from_collection',
    'FeatureTransformProcessor',
    'aggregate_across_splits',
    'select_final_transforms',
    'apply_transformations_direct'
]