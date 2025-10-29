"""
Transformation Functions

Implements various normalization transformations and the Pearson P/DF metric.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional


# Available transformation types
TRANSFORM_TYPES = ['identity', 'log', 'sqrt', 'arcsinh', 'box_cox', 'yeo_johnson']


def fit_transform_params(data: np.ndarray, transform_type: str) -> Optional[Dict]:
    """
    Fit transformation parameters on training data.
    
    Args:
        data: Training data array
        transform_type: Type of transformation to fit
        
    Returns:
        Dictionary with transformation parameters, or None if fitting failed
    """
    # Remove non-finite values
    data = data[np.isfinite(data)]
    
    # Need minimum sample size
    if len(data) < 20:
        return None
    
    try:
        if transform_type == 'identity':
            return {'type': 'identity'}
        
        elif transform_type == 'log':
            min_val = np.min(data)
            offset = 0.0 if min_val > 0 else abs(min_val) + 1.0
            # Test transformation
            _ = np.log(data + offset)
            return {'type': 'log', 'offset': float(offset)}
        
        elif transform_type == 'sqrt':
            min_val = np.min(data)
            offset = 0.0 if min_val >= 0 else abs(min_val) + 0.1
            # Test transformation
            _ = np.sqrt(data + offset)
            return {'type': 'sqrt', 'offset': float(offset)}
        
        elif transform_type == 'arcsinh':
            # No parameters needed, test transformation
            _ = np.arcsinh(data)
            return {'type': 'arcsinh'}
        
        elif transform_type == 'box_cox':
            # Box-Cox requires strictly positive data
            if np.min(data) <= 0:
                return None
            
            # Suppress overflow warnings from scipy
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                try:
                    _, lmbda = stats.boxcox(data)
                    if not np.isfinite(lmbda):
                        return None
                    # Test transformation works
                    test_transformed = stats.boxcox(data, lmbda=lmbda)
                    if not np.all(np.isfinite(test_transformed)):
                        return None
                    return {'type': 'box_cox', 'lambda': float(lmbda)}
                except (ValueError, RuntimeError):
                    return None
        
        elif transform_type == 'yeo_johnson':
            # Yeo-Johnson works with any data
            # Suppress overflow warnings from scipy
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                try:
                    _, lmbda = stats.yeojohnson(data)
                    if not np.isfinite(lmbda):
                        return None
                    # Test transformation works
                    test_transformed = stats.yeojohnson(data, lmbda=lmbda)
                    if not np.all(np.isfinite(test_transformed)):
                        return None
                    return {'type': 'yeo_johnson', 'lambda': float(lmbda)}
                except (ValueError, RuntimeError):
                    return None
        
        else:
            return None
            
    except Exception as e:
        # Transformation failed
        return None


def apply_transform(data: np.ndarray, transform_params: Dict) -> np.ndarray:
    """
    Apply fitted transformation to data.
    
    Args:
        data: Data to transform
        transform_params: Fitted transformation parameters
        
    Returns:
        Transformed data array
    """
    transform_type = transform_params['type']
    
    # Suppress overflow warnings
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        try:
            if transform_type == 'identity':
                return data
            
            elif transform_type == 'log':
                return np.log(data + transform_params['offset'])
            
            elif transform_type == 'sqrt':
                return np.sqrt(data + transform_params['offset'])
            
            elif transform_type == 'arcsinh':
                return np.arcsinh(data)
            
            elif transform_type == 'box_cox':
                return stats.boxcox(data, lmbda=transform_params['lambda'])
            
            elif transform_type == 'yeo_johnson':
                return stats.yeojohnson(data, lmbda=transform_params['lambda'])
            
            else:
                return data
                
        except Exception as e:
            # Return NaN array if transformation fails
            return np.full_like(data, np.nan)


def compute_pearson_p_df(data: np.ndarray) -> float:
    """
    Compute Pearson P/DF statistic (goodness of fit / degrees of freedom).
    
    Lower values indicate better normality. Values close to 1.0 or below
    indicate good approximation to normal distribution.
    
    Args:
        data: Transformed data to evaluate
        
    Returns:
        Pearson P/DF statistic (chi-square / df)
    """
    # Remove non-finite values
    data = data[np.isfinite(data)]
    
    # Need minimum sample size
    if len(data) < 20:
        return 999.0
    
    n = len(data)
    
    # Choose number of bins based on sample size
    if n < 100:
        k = 5
    elif n < 500:
        k = 10
    else:
        k = 20
    
    # Standardize data
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0 or not np.isfinite(std):
        return 999.0
    
    z_scores = (data - mean) / std
    
    # Create bins with equal probability under normal distribution
    bin_edges = stats.norm.ppf(np.linspace(0, 1, k + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Count observed frequencies
    observed, _ = np.histogram(z_scores, bins=bin_edges)
    
    # Expected frequencies (equal for each bin)
    expected = np.full(k, n / k)
    
    # Compute chi-square statistic
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    
    # Degrees of freedom (k bins - 1 - 2 estimated parameters)
    df = k - 1 - 2
    
    if df <= 0:
        return 999.0
    
    # Return P/DF ratio
    p_df = chi2_stat / df
    
    return float(p_df)