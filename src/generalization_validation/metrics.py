"""Statistical metrics for generalization validation."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from collections import Counter


def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', gamma: float = None, max_samples: int = 5000) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.

    Args:
        X: (n_samples, n_features) array
        Y: (m_samples, n_features) array
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel bandwidth (default: 1/n_features)
        max_samples: Maximum samples to use (subsample if exceeded to avoid memory issues)

    Returns:
        MMD value
    """
    # Subsample if too many samples (avoid memory issues)
    if X.shape[0] > max_samples:
        indices_X = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices_X]

    if Y.shape[0] > max_samples:
        indices_Y = np.random.choice(Y.shape[0], max_samples, replace=False)
        Y = Y[indices_Y]

    n = X.shape[0]
    m = Y.shape[0]

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    if kernel == 'rbf':
        # Compute kernel matrices
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
    elif kernel == 'linear':
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
    mmd_squared = (XX.sum() - np.trace(XX)) / (n * (n - 1))
    mmd_squared += (YY.sum() - np.trace(YY)) / (m * (m - 1))
    mmd_squared -= 2 * XY.sum() / (n * m)

    return np.sqrt(max(mmd_squared, 0))


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix.

    Args:
        X: (n, d) array
        Y: (m, d) array
        gamma: Kernel bandwidth

    Returns:
        K: (n, m) kernel matrix
    """
    # Compute pairwise squared distances
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
    dist_squared = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    # Apply RBF kernel
    K = np.exp(-gamma * dist_squared)

    return K


def compute_ks_tests(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    Compute Kolmogorov-Smirnov tests for each dimension.

    Args:
        X: (n_samples, n_features) array
        Y: (m_samples, n_features) array

    Returns:
        Dictionary with KS statistics and p-values
    """
    n_features = X.shape[1]

    ks_statistics = []
    p_values = []
    rejected_dims = []

    for i in range(n_features):
        statistic, p_value = stats.ks_2samp(X[:, i], Y[:, i])
        ks_statistics.append(statistic)
        p_values.append(p_value)

        # Check if null hypothesis rejected at alpha=0.05
        if p_value < 0.05:
            rejected_dims.append(i)

    return {
        'mean_ks_statistic': np.mean(ks_statistics),
        'max_ks_statistic': np.max(ks_statistics),
        'mean_p_value': np.mean(p_values),
        'min_p_value': np.min(p_values),
        'rejection_rate': len(rejected_dims) / n_features,
        'rejected_dimensions': rejected_dims,
        'all_statistics': ks_statistics,
        'all_p_values': p_values
    }


def compute_correlation_distance(X: np.ndarray, Y: np.ndarray, use_clr: bool = True) -> Dict[str, float]:
    """
    Compute distance between correlation matrices.

    For compositional/normalized data (e.g., probability distributions),
    uses Centered Log-Ratio (CLR) transformation to handle the sum-to-one constraint.

    Args:
        X: (n_samples, n_features) array
        Y: (m_samples, n_features) array
        use_clr: Whether to use CLR transformation for compositional data (default: True)

    Returns:
        Dictionary with correlation distances
    """
    import warnings

    # For compositional data (probability distributions), apply CLR transformation
    # This removes the sum-to-one constraint and allows proper correlation computation
    if use_clr:
        X_transformed = _clr_transform(X)
        Y_transformed = _clr_transform(Y)
    else:
        X_transformed = X
        Y_transformed = Y

    # Suppress numpy warning about division by zero in corrcoef when features have zero variance
    # This is expected for constant features (e.g., LOB bins with no variation)
    # We handle NaNs properly by replacing them with 0 after computation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in divide')
        warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')

        # Compute correlation matrices
        corr_X = np.corrcoef(X_transformed, rowvar=False)
        corr_Y = np.corrcoef(Y_transformed, rowvar=False)

    # Replace NaNs with 0 (happens when feature is constant/zero variance)
    corr_X = np.nan_to_num(corr_X, nan=0.0)
    corr_Y = np.nan_to_num(corr_Y, nan=0.0)

    # Frobenius norm (unbounded, for backward compatibility)
    frobenius_norm = np.linalg.norm(corr_X - corr_Y, ord='fro')

    # Frobenius correlation (bounded [-1, 1], more interpretable)
    # Treats correlation matrices as vectors and computes their correlation
    norm_X = np.linalg.norm(corr_X, ord='fro')
    norm_Y = np.linalg.norm(corr_Y, ord='fro')
    if norm_X > 0 and norm_Y > 0:
        frobenius_corr = np.trace(corr_X.T @ corr_Y) / (norm_X * norm_Y)
    else:
        frobenius_corr = 0.0

    # Mean absolute difference (bounded [0, 2])
    mad = np.mean(np.abs(corr_X - corr_Y))

    # Max absolute difference (bounded [0, 2])
    max_diff = np.max(np.abs(corr_X - corr_Y))

    return {
        'frobenius_norm': frobenius_norm,
        'frobenius_correlation': frobenius_corr,
        'mean_absolute_diff': mad,
        'max_absolute_diff': max_diff,
        'corr_original': corr_X,
        'corr_synthetic': corr_Y
    }


def _clr_transform(X: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Apply Centered Log-Ratio (CLR) transformation for compositional data.

    The CLR transformation removes the sum-to-one constraint, allowing
    proper correlation analysis of compositional data like probability distributions.

    CLR(x) = log(x / geometric_mean(x))

    Args:
        X: (n_samples, n_features) array of compositional data
        epsilon: Small constant to avoid log(0) (default: 1e-10)

    Returns:
        X_clr: (n_samples, n_features) CLR-transformed array
    """
    # Add small epsilon to avoid log(0)
    X_safe = X + epsilon

    # Compute geometric mean for each sample
    # geometric_mean = exp(mean(log(x)))
    log_X = np.log(X_safe)
    geometric_mean = np.exp(np.mean(log_X, axis=1, keepdims=True))

    # Apply CLR transformation
    X_clr = np.log(X_safe / geometric_mean)

    return X_clr


def compute_transition_matrix(sequences: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Compute transition probability matrix from code sequences.

    Args:
        sequences: (n_sequences, seq_len) array of code indices
        vocab_size: Number of unique codes

    Returns:
        transition_matrix: (vocab_size, vocab_size) array of transition probabilities
    """
    # Initialize count matrix
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    # Count transitions
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_code = seq[i]
            to_code = seq[i + 1]
            counts[from_code, to_code] += 1

    # Normalize to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

    transition_matrix = counts / row_sums

    return transition_matrix


def extract_ngrams(sequences: np.ndarray, n: int) -> Counter:
    """
    Extract n-grams from sequences.

    Args:
        sequences: (n_sequences, seq_len) array of code indices
        n: N-gram size (e.g., 2 for bigrams, 3 for trigrams)

    Returns:
        Counter of n-grams
    """
    ngrams = []

    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i + n])
            ngrams.append(ngram)

    return Counter(ngrams)


def compare_ngrams(
    ngrams_val: Counter,
    ngrams_syn: Counter,
    top_k: int = 20
) -> Dict:
    """
    Compare n-gram distributions.

    Args:
        ngrams_val: N-grams from validation data
        ngrams_syn: N-grams from synthetic data
        top_k: Number of top n-grams to compare

    Returns:
        Dictionary with comparison metrics
    """
    # Get top-k n-grams from validation
    top_val = ngrams_val.most_common(top_k)
    top_syn = ngrams_syn.most_common(top_k)

    # Extract n-grams and frequencies
    val_ngrams = set([ng for ng, _ in top_val])
    syn_ngrams = set([ng for ng, _ in top_syn])

    # Compute overlap
    overlap = len(val_ngrams & syn_ngrams)
    overlap_ratio = overlap / top_k

    # Compute frequency correlation for common n-grams
    common = val_ngrams & syn_ngrams
    if len(common) > 1:
        val_freqs = [ngrams_val[ng] for ng in common]
        syn_freqs = [ngrams_syn[ng] for ng in common]

        # Check for zero variance (all values are the same)
        if np.std(val_freqs) > 0 and np.std(syn_freqs) > 0:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered')
                warnings.filterwarnings('ignore', message='divide by zero encountered')
                freq_corr = np.corrcoef(val_freqs, syn_freqs)[0, 1]
                freq_corr = 0.0 if np.isnan(freq_corr) else freq_corr
        else:
            freq_corr = 0.0
    else:
        freq_corr = 0.0

    return {
        'top_k': top_k,
        'overlap_count': overlap,
        'overlap_ratio': overlap_ratio,
        'frequency_correlation': freq_corr,
        'top_val_ngrams': top_val,
        'top_syn_ngrams': top_syn
    }


def compute_code_frequency(codes: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Compute frequency of each code.

    Args:
        codes: Array of code indices
        vocab_size: Total vocabulary size

    Returns:
        frequencies: (vocab_size,) array of frequencies
    """
    counts = np.bincount(codes.flatten(), minlength=vocab_size)
    frequencies = counts / counts.sum()
    return frequencies


def jensen_shannon_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        P: Probability distribution
        Q: Probability distribution

    Returns:
        JS divergence
    """
    # Ensure no zeros
    P = P + 1e-10
    Q = Q + 1e-10

    # Normalize
    P = P / P.sum()
    Q = Q / Q.sum()

    # Compute M
    M = 0.5 * (P + Q)

    # KL divergences
    kl_pm = np.sum(P * np.log(P / M))
    kl_qm = np.sum(Q * np.log(Q / M))

    # JS divergence
    js = 0.5 * (kl_pm + kl_qm)

    return js


def compute_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    Compute cosine similarity between corresponding samples.

    Args:
        X: (n_samples, n_features) array
        Y: (n_samples, n_features) array

    Returns:
        Dictionary with cosine similarity statistics
    """
    # Compute per-sample cosine similarity
    # Normalize each row
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)

    # Compute dot product for each sample
    per_sample_similarity = np.sum(X_norm * Y_norm, axis=1)

    return {
        'mean_cosine_similarity': float(np.mean(per_sample_similarity)),
        'std_cosine_similarity': float(np.std(per_sample_similarity)),
        'min_cosine_similarity': float(np.min(per_sample_similarity)),
        'max_cosine_similarity': float(np.max(per_sample_similarity)),
        'median_cosine_similarity': float(np.median(per_sample_similarity)),
        'per_sample_similarities': per_sample_similarity
    }


# =================================================================================================
# Target Field Validation Metrics
# =================================================================================================

def compare_target_distributions(
    val_targets: np.ndarray,
    syn_targets: np.ndarray,
    n_bins: int = 50
) -> Dict[str, float]:
    """
    Compare target value distributions between validation and synthetic data.

    Args:
        val_targets: (N,) validation target values
        syn_targets: (M,) synthetic target values
        n_bins: Number of bins for histogram-based JS divergence

    Returns:
        Dictionary with distribution comparison metrics
    """
    # 1. Basic statistics comparison
    val_mean = float(np.mean(val_targets))
    syn_mean = float(np.mean(syn_targets))
    val_std = float(np.std(val_targets))
    syn_std = float(np.std(syn_targets))
    val_min = float(np.min(val_targets))
    syn_min = float(np.min(syn_targets))
    val_max = float(np.max(val_targets))
    syn_max = float(np.max(val_targets))

    mean_diff = float(np.abs(val_mean - syn_mean))
    std_ratio = float(syn_std / (val_std + 1e-10))

    # 2. Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = stats.ks_2samp(val_targets, syn_targets)

    # 3. Wasserstein distance (Earth Mover's Distance)
    wasserstein = float(stats.wasserstein_distance(val_targets, syn_targets))

    # 4. JS divergence (histogram-based)
    # Determine common bin edges
    all_targets = np.concatenate([val_targets, syn_targets])
    bin_edges = np.linspace(all_targets.min(), all_targets.max(), n_bins + 1)

    val_hist, _ = np.histogram(val_targets, bins=bin_edges)
    syn_hist, _ = np.histogram(syn_targets, bins=bin_edges)

    # Convert to probabilities
    val_prob = (val_hist + 1e-10) / (val_hist.sum() + n_bins * 1e-10)
    syn_prob = (syn_hist + 1e-10) / (syn_hist.sum() + n_bins * 1e-10)

    js_div = jensen_shannon_divergence(val_prob, syn_prob)

    return {
        'val_mean': val_mean,
        'syn_mean': syn_mean,
        'val_std': val_std,
        'syn_std': syn_std,
        'val_min': val_min,
        'syn_min': syn_min,
        'val_max': val_max,
        'syn_max': syn_max,
        'mean_abs_diff': mean_diff,
        'std_ratio': std_ratio,
        'ks_statistic': float(ks_statistic),
        'ks_p_value': float(ks_p_value),
        'wasserstein_distance': wasserstein,
        'js_divergence': float(js_div)
    }


def compute_autocorrelation(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Compute autocorrelation function.

    Args:
        x: (N,) time series
        max_lag: Maximum lag to compute

    Returns:
        acf: (max_lag+1,) autocorrelations for lags 0 to max_lag
    """
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for k in range(1, max_lag + 1):
        c_k = np.dot(x[:-k], x[k:]) / len(x)
        acf[k] = c_k / c0

    return acf


def compare_target_autocorrelation(
    val_targets: np.ndarray,
    syn_targets: np.ndarray,
    max_lag: int = 20
) -> Dict:
    """
    Compare autocorrelation structure of target sequences.

    Args:
        val_targets: (N,) validation targets
        syn_targets: (M,) synthetic targets
        max_lag: Maximum lag to analyze

    Returns:
        Dictionary with ACF comparison metrics
    """
    # Compute ACF for both
    acf_val = compute_autocorrelation(val_targets, max_lag)
    acf_syn = compute_autocorrelation(syn_targets, max_lag)

    # Mean absolute error
    acf_mae = float(np.mean(np.abs(acf_val - acf_syn)))

    # Correlation between ACF curves (excluding lag 0 which is always 1)
    if max_lag > 1:
        acf_corr = float(np.corrcoef(acf_val[1:], acf_syn[1:])[0, 1])
        if np.isnan(acf_corr):
            acf_corr = 0.0
    else:
        acf_corr = 0.0

    return {
        'acf_val': acf_val.tolist(),
        'acf_syn': acf_syn.tolist(),
        'acf_mae': acf_mae,
        'acf_correlation': acf_corr,
        'max_lag': max_lag
    }


def compare_volatility_clustering(
    val_targets: np.ndarray,
    syn_targets: np.ndarray,
    max_lag: int = 20
) -> Dict:
    """
    Compare volatility clustering (autocorrelation of absolute values).

    This tests whether the model captures GARCH-like effects where
    large returns tend to be followed by large returns.

    Args:
        val_targets: (N,) validation targets
        syn_targets: (M,) synthetic targets
        max_lag: Maximum lag to analyze

    Returns:
        Dictionary with volatility clustering metrics
    """
    # Compute ACF of absolute values
    abs_val = np.abs(val_targets)
    abs_syn = np.abs(syn_targets)

    acf_abs_val = compute_autocorrelation(abs_val, max_lag)
    acf_abs_syn = compute_autocorrelation(abs_syn, max_lag)

    # Mean absolute error
    vol_clustering_mae = float(np.mean(np.abs(acf_abs_val - acf_abs_syn)))

    # Correlation between absolute ACF curves
    if max_lag > 1:
        vol_clustering_corr = float(np.corrcoef(acf_abs_val[1:], acf_abs_syn[1:])[0, 1])
        if np.isnan(vol_clustering_corr):
            vol_clustering_corr = 0.0
    else:
        vol_clustering_corr = 0.0

    return {
        'acf_abs_val': acf_abs_val.tolist(),
        'acf_abs_syn': acf_abs_syn.tolist(),
        'vol_clustering_mae': vol_clustering_mae,
        'vol_clustering_correlation': vol_clustering_corr,
        'max_lag': max_lag
    }
