"""
EWMA Scaler Module

Implements EWMA-based standardization with different half-life parameters.
Provides methods for fitting on training data and applying to new data.
"""

import numpy as np
from typing import Dict, Optional, List


class EWMAScaler:
    """
    EWMA-based standardization scaler.
    
    Maintains exponentially weighted moving average (EWMA) of mean and variance
    for online standardization of streaming data.
    """
    
    def __init__(self, half_life: int):
        """
        Initialize EWMA scaler.
        
        Args:
            half_life: Half-life parameter for EWMA (in samples)
        """
        self.half_life = half_life
        self.alpha = 1.0 - (0.5 ** (1.0 / float(half_life)))
        
        # EWMA state
        self.ewma_mean = None
        self.ewma_var = None
        self.initialized = False
        self.n_samples = 0
    
    def update(self, value: float):
        """
        Update EWMA statistics with a new value.
        
        Args:
            value: New value to incorporate into EWMA
        """
        if not np.isfinite(value):
            return
        
        if not self.initialized:
            # Initialize with first value
            self.ewma_mean = value
            self.ewma_var = 0.0
            self.initialized = True
        else:
            # Update EWMA mean
            delta = value - self.ewma_mean
            self.ewma_mean += self.alpha * delta
            
            # Update EWMA variance (Welford's algorithm adapted for EWMA)
            self.ewma_var = (1 - self.alpha) * (self.ewma_var + self.alpha * delta * delta)
        
        self.n_samples += 1
    
    def fit(self, data: np.ndarray):
        """
        Fit the scaler on training data.
        
        Args:
            data: Training data array
        """
        # Reset state
        self.ewma_mean = None
        self.ewma_var = None
        self.initialized = False
        self.n_samples = 0
        
        # Update with each value in sequence
        for value in data:
            self.update(value)
    
    def standardize(self, value: float, clip_std: float = 3.0) -> float:
        """
        Standardize a value using current EWMA statistics.
        
        Args:
            value: Value to standardize
            clip_std: Number of standard deviations to clip outliers
            
        Returns:
            Standardized value
        """
        if not self.initialized:
            return value
        
        if not np.isfinite(value):
            return value
        
        # Compute standard deviation (add small epsilon for numerical stability)
        std = np.sqrt(self.ewma_var + 1e-12)
        
        # Standardize
        z_score = (value - self.ewma_mean) / std
        
        # Clip outliers
        z_score = np.clip(z_score, -clip_std, clip_std)
        
        return float(z_score)
    
    def transform(self, data: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
        """
        Transform an array of values using current EWMA statistics.
        
        Note: This applies the SAME statistics to all values (batch transform).
        Use update() + standardize() for sequential processing.
        
        Args:
            data: Data array to transform
            clip_std: Number of standard deviations to clip outliers
            
        Returns:
            Transformed data array
        """
        if not self.initialized:
            return data
        
        std = np.sqrt(self.ewma_var + 1e-12)
        z_scores = (data - self.ewma_mean) / std
        z_scores = np.clip(z_scores, -clip_std, clip_std)
        
        return z_scores
    
    def get_params(self) -> Dict:
        """
        Get scaler parameters.
        
        Returns:
            Dictionary with scaler parameters
        """
        return {
            'half_life': self.half_life,
            'alpha': self.alpha,
            'ewma_mean': float(self.ewma_mean) if self.ewma_mean is not None else None,
            'ewma_var': float(self.ewma_var) if self.ewma_var is not None else None,
            'n_samples': self.n_samples
        }


def fit_ewma_scaler(data: np.ndarray, half_life: int) -> Optional[EWMAScaler]:
    """
    Fit an EWMA scaler on training data.
    
    Args:
        data: Training data array
        half_life: Half-life parameter for EWMA
        
    Returns:
        Fitted EWMAScaler, or None if fitting failed
    """
    # Remove non-finite values
    data = data[np.isfinite(data)]
    
    # Need minimum sample size
    if len(data) < 20:
        return None
    
    try:
        scaler = EWMAScaler(half_life)
        scaler.fit(data)
        
        # Verify scaler is valid
        if not scaler.initialized:
            return None
        
        # Test that it can transform data
        test_transformed = scaler.transform(data[:10])
        if not np.all(np.isfinite(test_transformed)):
            return None
        
        return scaler
        
    except Exception:
        return None


def apply_ewma_standardization(
    data: np.ndarray, 
    scaler: EWMAScaler, 
    clip_std: float = 3.0
) -> np.ndarray:
    """
    Apply EWMA standardization to data.
    
    This is for batch transformation where the scaler state is fixed.
    For sequential processing, use scaler.update() + scaler.standardize().
    
    Args:
        data: Data to standardize
        scaler: Fitted EWMA scaler
        clip_std: Number of standard deviations to clip
        
    Returns:
        Standardized data array
    """
    return scaler.transform(data, clip_std=clip_std)


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
    from scipy import stats
    
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


# Available half-life candidates to test
HALFLIFE_CANDIDATES = [5, 10, 20, 40, 80]