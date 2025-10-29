"""
EWMA Standardization Module

Two-stage EWMA standardization pipeline:
- Stage 10: Select optimal half-life parameters using CPCV validation
- Stage 11: Apply EWMA standardization with selected parameters

This module provides split-aware EWMA standardization that respects
embargo boundaries in combinatorial purged cross-validation.
"""

from .ewma_scaler import (
    EWMAScaler,
    fit_ewma_scaler,
    apply_ewma_standardization,
    compute_pearson_p_df,
    HALFLIFE_CANDIDATES
)

from .processor import EWMAHalfLifeProcessor

from .data_loader import (
    get_all_hours,
    load_hour_batch,
    identify_feature_names,
    filter_standardizable_features
)

from .aggregator import (
    aggregate_across_splits,
    select_final_half_lives
)

from .apply_scaler import EWMAStandardizationApplicator

from .mlflow_logger import (
    log_split_results,
    log_aggregated_results
)

__all__ = [
    # Scaler
    'EWMAScaler',
    'fit_ewma_scaler',
    'apply_ewma_standardization',
    'compute_pearson_p_df',
    'HALFLIFE_CANDIDATES',
    
    # Processor
    'EWMAHalfLifeProcessor',
    
    # Data loading
    'get_all_hours',
    'load_hour_batch',
    'identify_feature_names',
    'filter_standardizable_features',
    
    # Aggregation
    'aggregate_across_splits',
    'select_final_half_lives',
    
    # Application
    'EWMAStandardizationApplicator',
    
    # Logging
    'log_split_results',
    'log_aggregated_results'
]