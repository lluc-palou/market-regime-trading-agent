"""Statistical metrics for generalization validation."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from collections import Counter


def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', gamma: float = None) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.

    Args:
        X: (n_samples, n_features) array
        Y: (m_samples, n_features) array
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel bandwidth (default: 1/n_features)

    Returns:
        MMD value
    """
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


def compute_correlation_distance(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    Compute distance between correlation matrices.

    Args:
        X: (n_samples, n_features) array
        Y: (m_samples, n_features) array

    Returns:
        Dictionary with correlation distances
    """
    import warnings

    # Suppress numpy warning about division by zero in corrcoef when features have zero variance
    # This is expected for constant features (e.g., LOB bins with no variation)
    # We handle NaNs properly by replacing them with 0 after computation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in divide')

        # Compute correlation matrices
        corr_X = np.corrcoef(X, rowvar=False)
        corr_Y = np.corrcoef(Y, rowvar=False)

    # Replace NaNs with 0 (happens when feature is constant/zero variance)
    corr_X = np.nan_to_num(corr_X, nan=0.0)
    corr_Y = np.nan_to_num(corr_Y, nan=0.0)

    # Frobenius norm
    frobenius = np.linalg.norm(corr_X - corr_Y, ord='fro')

    # Mean absolute difference
    mad = np.mean(np.abs(corr_X - corr_Y))

    # Max absolute difference
    max_diff = np.max(np.abs(corr_X - corr_Y))

    return {
        'frobenius_norm': frobenius,
        'mean_absolute_diff': mad,
        'max_absolute_diff': max_diff,
        'corr_original': corr_X,
        'corr_synthetic': corr_Y
    }


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
    if len(common) > 0:
        val_freqs = [ngrams_val[ng] for ng in common]
        syn_freqs = [ngrams_syn[ng] for ng in common]
        freq_corr = np.corrcoef(val_freqs, syn_freqs)[0, 1]
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
