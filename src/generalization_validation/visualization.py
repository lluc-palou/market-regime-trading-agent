"""Visualization utilities for generalization validation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots without display
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional, Tuple

# Set style for publication-quality plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Colorblind-friendly palette
COLORS = {
    'significant': '#D55E00',  # Orange - deviates from normal
    'non_significant': '#0072B2',  # Blue - consistent with normal
    'reference': '#000000',  # Black for reference lines
    'ci': '#000000'  # Black for CI lines
}


def plot_umap_comparison(
    X_original: np.ndarray,
    X_synthetic: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    method: str = 'umap',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: int = 30,
    label_second: str = 'Synthetic'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot UMAP or t-SNE comparison of original vs synthetic data.

    Args:
        X_original: (n, d) array of original data
        X_synthetic: (m, d) array of synthetic data
        title: Plot title
        save_path: Path to save figure
        method: 'umap' or 'tsne'
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        perplexity: t-SNE perplexity parameter
        label_second: Label for second dataset (default: 'Synthetic')

    Returns:
        Tuple of (original_embedding, synthetic_embedding)
    """
    # Combine data
    X_combined = np.vstack([X_original, X_synthetic])
    labels = np.array(['Original'] * len(X_original) + [label_second] * len(X_synthetic))

    # Dimensionality reduction
    if method == 'umap':
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_jobs=1)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    embedding = reducer.fit_transform(X_combined)

    # Split back
    n_orig = len(X_original)
    embedding_orig = embedding[:n_orig]
    embedding_syn = embedding[n_orig:]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter_orig = ax.scatter(
        embedding_orig[:, 0], embedding_orig[:, 1],
        c=COLORS['non_significant'], alpha=0.3, s=10, label='Original', rasterized=True
    )
    scatter_syn = ax.scatter(
        embedding_syn[:, 0], embedding_syn[:, 1],
        c=COLORS['significant'], alpha=0.3, s=10, label=label_second, rasterized=True
    )

    ax.set_xlabel(f'{method.upper()} Dimension 1', color='black', fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Dimension 2', color='black', fontweight='bold')
    ax.set_title(title, color='black', fontweight='bold', pad=15)
    ax.legend()
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return embedding_orig, embedding_syn


def plot_reconstruction_error(
    reconstruction_errors: np.ndarray,
    save_path: Optional[Path] = None,
    bins: int = 50
):
    """
    Plot per-feature reconstruction error.

    Args:
        reconstruction_errors: (n_features,) array of MSE per feature
        save_path: Path to save figure
        bins: Number of bins for histogram
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Bar plot of errors
    axes[0].bar(range(len(reconstruction_errors)), reconstruction_errors, width=1.0, color=COLORS['non_significant'])
    axes[0].set_xlabel('Feature Index', color='black', fontweight='bold')
    axes[0].set_ylabel('MSE', color='black', fontweight='bold')
    axes[0].set_title('Per-Feature Reconstruction Error', color='black', fontweight='bold', pad=15)
    axes[0].tick_params(colors='black')
    for spine in axes[0].spines.values():
        spine.set_color('black')

    # Histogram of errors
    axes[1].hist(reconstruction_errors, bins=bins, color=COLORS['non_significant'], edgecolor='black')
    axes[1].set_xlabel('MSE', color='black', fontweight='bold')
    axes[1].set_ylabel('Count', color='black', fontweight='bold')
    axes[1].set_title('Distribution of Reconstruction Errors', color='black', fontweight='bold', pad=15)
    axes[1].tick_params(colors='black')
    for spine in axes[1].spines.values():
        spine.set_color('black')

    # Add statistics
    mean_err = np.mean(reconstruction_errors)
    median_err = np.median(reconstruction_errors)
    max_err = np.max(reconstruction_errors)
    axes[1].axvline(mean_err, color=COLORS['significant'], linestyle='--', linewidth=2, label=f'Mean: {mean_err:.4f}')
    axes[1].axvline(median_err, color=COLORS['reference'], linestyle='--', linewidth=2, label=f'Median: {median_err:.4f}')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_code_frequency(
    freq_val: np.ndarray,
    freq_syn: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot code frequency comparison.

    Args:
        freq_val: Validation code frequencies
        freq_syn: Synthetic code frequencies
        save_path: Path to save figure
    """
    vocab_size = len(freq_val)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Overlaid histograms
    x = np.arange(vocab_size)
    width = 0.4

    axes[0].bar(x - width/2, freq_val, width, label='Validation', alpha=0.7, color=COLORS['non_significant'])
    axes[0].bar(x + width/2, freq_syn, width, label='Synthetic', alpha=0.7, color=COLORS['significant'])
    axes[0].set_xlabel('Code Index', color='black', fontweight='bold')
    axes[0].set_ylabel('Frequency', color='black', fontweight='bold')
    axes[0].set_title('Codebook Index Empirical Distribution', color='black', fontweight='bold', pad=15)
    axes[0].legend()
    axes[0].tick_params(colors='black')
    for spine in axes[0].spines.values():
        spine.set_color('black')

    # Scatter plot for correlation
    axes[1].scatter(freq_val, freq_syn, alpha=0.5, s=30, color=COLORS['non_significant'])
    axes[1].plot([0, freq_val.max()], [0, freq_val.max()], color=COLORS['reference'], linestyle='--', linewidth=2, label='y=x')
    axes[1].set_xlabel('Validation Frequency', color='black', fontweight='bold')
    axes[1].set_ylabel('Synthetic Frequency', color='black', fontweight='bold')
    axes[1].set_title('Code Frequency Correlation', color='black', fontweight='bold', pad=15)
    axes[1].legend()
    axes[1].tick_params(colors='black')
    for spine in axes[1].spines.values():
        spine.set_color('black')

    # Compute correlation
    corr = np.corrcoef(freq_val, freq_syn)[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr:.4f}',
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_transition_matrix(
    trans_val: np.ndarray,
    trans_syn: np.ndarray,
    save_path: Optional[Path] = None,
    vmax: float = None
):
    """
    Plot transition matrix comparison.

    Args:
        trans_val: Validation transition matrix
        trans_syn: Synthetic transition matrix
        save_path: Path to save figure
        vmax: Maximum value for color scale
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if vmax is None:
        vmax = max(trans_val.max(), trans_syn.max())

    # Validation transition matrix
    sns.heatmap(trans_val, ax=axes[0], cmap='Blues', vmin=0, vmax=vmax, cbar=True,
                cbar_kws={'label': ''})
    axes[0].set_title('Validation Transition Matrix', color='black', fontweight='bold', pad=15)
    axes[0].set_xlabel('To Code', color='black', fontweight='bold')
    axes[0].set_ylabel('From Code', color='black', fontweight='bold')
    axes[0].tick_params(colors='black')
    for spine in axes[0].spines.values():
        spine.set_color('black')

    # Synthetic transition matrix
    sns.heatmap(trans_syn, ax=axes[1], cmap='Oranges', vmin=0, vmax=vmax, cbar=True,
                cbar_kws={'label': ''})
    axes[1].set_title('Synthetic Transition Matrix', color='black', fontweight='bold', pad=15)
    axes[1].set_xlabel('To Code', color='black', fontweight='bold')
    axes[1].set_ylabel('From Code', color='black', fontweight='bold')
    axes[1].tick_params(colors='black')
    for spine in axes[1].spines.values():
        spine.set_color('black')

    # Difference matrix
    diff = np.abs(trans_val - trans_syn)
    sns.heatmap(diff, ax=axes[2], cmap='Greys', vmin=0, cbar=True,
                cbar_kws={'label': ''})
    axes[2].set_title('Absolute Difference', color='black', fontweight='bold', pad=15)
    axes[2].set_xlabel('To Code', color='black', fontweight='bold')
    axes[2].set_ylabel('From Code', color='black', fontweight='bold')
    axes[2].tick_params(colors='black')
    for spine in axes[2].spines.values():
        spine.set_color('black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ngram_comparison(
    top_val: list,
    top_syn: list,
    n: int,
    save_path: Optional[Path] = None
):
    """
    Plot top n-grams comparison.

    Args:
        top_val: List of (ngram, count) for validation
        top_syn: List of (ngram, count) for synthetic
        n: N-gram size
        save_path: Path to save figure
    """
    k = min(len(top_val), len(top_syn), 20)

    # Convert n to descriptive name
    ngram_name = {2: 'Bigrams', 3: 'Trigrams'}.get(n, f'{n}-grams')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Validation - format ngrams as tuples of ints
    ngrams_val = [str(tuple(int(x) for x in ng)) for ng, _ in top_val[:k]]
    counts_val = [cnt for _, cnt in top_val[:k]]

    axes[0].barh(range(k), counts_val, tick_label=ngrams_val, color=COLORS['non_significant'])
    axes[0].set_xlabel('Count', color='black', fontweight='bold')
    axes[0].set_title(f'Validation {ngram_name}', color='black', fontweight='bold', pad=15)
    axes[0].invert_yaxis()
    axes[0].tick_params(colors='black')
    for spine in axes[0].spines.values():
        spine.set_color('black')

    # Synthetic - format ngrams as tuples of ints
    ngrams_syn = [str(tuple(int(x) for x in ng)) for ng, _ in top_syn[:k]]
    counts_syn = [cnt for _, cnt in top_syn[:k]]

    axes[1].barh(range(k), counts_syn, tick_label=ngrams_syn, color=COLORS['significant'])
    axes[1].set_xlabel('Count', color='black', fontweight='bold')
    axes[1].set_title(f'Synthetic {ngram_name}', color='black', fontweight='bold', pad=15)
    axes[1].invert_yaxis()
    axes[1].tick_params(colors='black')
    for spine in axes[1].spines.values():
        spine.set_color('black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_matrices(
    corr_orig: np.ndarray,
    corr_syn: np.ndarray,
    save_path: Optional[Path] = None,
    sample_features: int = 100
):
    """
    Plot correlation matrix comparison (sampled for visualization).

    Args:
        corr_orig: Original correlation matrix
        corr_syn: Synthetic correlation matrix
        save_path: Path to save figure
        sample_features: Number of features to visualize
    """
    # Sample features if too many
    n_features = corr_orig.shape[0]
    if n_features > sample_features:
        indices = np.linspace(0, n_features - 1, sample_features, dtype=int)
        corr_orig_vis = corr_orig[np.ix_(indices, indices)]
        corr_syn_vis = corr_syn[np.ix_(indices, indices)]
    else:
        corr_orig_vis = corr_orig
        corr_syn_vis = corr_syn

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original
    sns.heatmap(corr_orig_vis, ax=axes[0], cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, cbar=True, cbar_kws={'label': ''})
    axes[0].set_title('Original Correlation', color='black', fontweight='bold', pad=15)
    axes[0].tick_params(colors='black')
    for spine in axes[0].spines.values():
        spine.set_color('black')

    # Synthetic
    sns.heatmap(corr_syn_vis, ax=axes[1], cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, cbar=True, cbar_kws={'label': ''})
    axes[1].set_title('Synthetic Correlation', color='black', fontweight='bold', pad=15)
    axes[1].tick_params(colors='black')
    for spine in axes[1].spines.values():
        spine.set_color('black')

    # Difference
    diff = np.abs(corr_orig_vis - corr_syn_vis)
    sns.heatmap(diff, ax=axes[2], cmap='Greys', vmin=0, square=True, cbar=True,
                cbar_kws={'label': ''})
    axes[2].set_title('Absolute Difference', color='black', fontweight='bold', pad=15)
    axes[2].tick_params(colors='black')
    for spine in axes[2].spines.values():
        spine.set_color('black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =================================================================================================
# Target Field Visualization
# =================================================================================================

def plot_target_distribution(
    val_targets: np.ndarray,
    syn_targets: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot distribution comparison of target values.

    Args:
        val_targets: Validation target values
        syn_targets: Synthetic target values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram + KDE
    ax = axes[0]
    ax.hist(val_targets, bins=50, alpha=0.5, color=COLORS['non_significant'],
            label='Validation', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(syn_targets, bins=50, alpha=0.5, color=COLORS['significant'],
            label='Synthetic', density=True, edgecolor='black', linewidth=0.5)

    # Add KDE
    from scipy.stats import gaussian_kde
    kde_val = gaussian_kde(val_targets)
    kde_syn = gaussian_kde(syn_targets)

    x_range = np.linspace(min(val_targets.min(), syn_targets.min()),
                          max(val_targets.max(), syn_targets.max()), 200)
    ax.plot(x_range, kde_val(x_range), color=COLORS['non_significant'],
            linewidth=2, linestyle='--', label='Validation KDE')
    ax.plot(x_range, kde_syn(x_range), color=COLORS['significant'],
            linewidth=2, linestyle='--', label='Synthetic KDE')

    ax.set_xlabel('Target Value', color='black', fontweight='bold')
    ax.set_ylabel('Density', color='black', fontweight='bold')
    ax.set_title('Target Distribution Comparison', color='black', fontweight='bold', pad=15)
    ax.legend()
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1]
    from scipy import stats
    stats.probplot(val_targets, dist="norm", plot=None)
    stats.probplot(syn_targets, dist="norm", plot=None)

    # Sort both for Q-Q plot
    val_sorted = np.sort(val_targets)
    syn_sorted = np.sort(syn_targets)

    # Interpolate to same length for fair comparison
    if len(val_sorted) != len(syn_sorted):
        common_quantiles = np.linspace(0, 1, min(len(val_sorted), len(syn_sorted)))
        val_quantiles = np.quantile(val_targets, common_quantiles)
        syn_quantiles = np.quantile(syn_targets, common_quantiles)
    else:
        val_quantiles = val_sorted
        syn_quantiles = syn_sorted

    ax.scatter(val_quantiles, syn_quantiles, alpha=0.5, s=10,
               color=COLORS['non_significant'], edgecolor='none')

    # Reference line
    lims = [min(val_quantiles.min(), syn_quantiles.min()),
            max(val_quantiles.max(), syn_quantiles.max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Perfect match')

    ax.set_xlabel('Validation Quantiles', color='black', fontweight='bold')
    ax.set_ylabel('Synthetic Quantiles', color='black', fontweight='bold')
    ax.set_title('Q-Q Plot', color='black', fontweight='bold', pad=15)
    ax.legend()
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_target_autocorrelation(
    acf_val: np.ndarray,
    acf_syn: np.ndarray,
    max_lag: int,
    save_path: Optional[Path] = None
):
    """
    Plot autocorrelation function comparison.

    Args:
        acf_val: Validation ACF
        acf_syn: Synthetic ACF
        max_lag: Maximum lag
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lags = np.arange(0, max_lag + 1)

    ax.plot(lags, acf_val, 'o-', color=COLORS['non_significant'],
            linewidth=2, markersize=6, label='Validation', alpha=0.8)
    ax.plot(lags, acf_syn, 's-', color=COLORS['significant'],
            linewidth=2, markersize=6, label='Synthetic', alpha=0.8)

    # Add 95% confidence interval for white noise
    n_samples = 1000  # Approximate for CI
    ci = 1.96 / np.sqrt(n_samples)
    ax.axhline(y=ci, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-ci, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Lag', color='black', fontweight='bold')
    ax.set_ylabel('Autocorrelation', color='black', fontweight='bold')
    ax.set_title('Target Autocorrelation Function', color='black', fontweight='bold', pad=15)
    ax.legend()
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_target_scatter(
    val_targets: np.ndarray,
    syn_targets: np.ndarray,
    save_path: Optional[Path] = None,
    max_points: int = 5000
):
    """
    Scatter plot comparing validation vs synthetic target values.

    Args:
        val_targets: Validation target values
        syn_targets: Synthetic target values
        save_path: Path to save figure
        max_points: Maximum points to plot (for performance)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Subsample if too many points
    if len(val_targets) > max_points:
        idx = np.random.choice(len(val_targets), max_points, replace=False)
        val_plot = val_targets[idx]
    else:
        val_plot = val_targets

    if len(syn_targets) > max_points:
        idx = np.random.choice(len(syn_targets), max_points, replace=False)
        syn_plot = syn_targets[idx]
    else:
        syn_plot = syn_targets

    # 2D histogram (heatmap)
    from matplotlib.colors import LogNorm
    h = ax.hist2d(val_plot, syn_plot, bins=50, cmap='Blues', norm=LogNorm(),
                  cmin=1, rasterized=True)
    plt.colorbar(h[3], ax=ax, label='Count (log scale)')

    # Reference line y=x
    lims = [min(val_targets.min(), syn_targets.min()),
            max(val_targets.max(), syn_targets.max())]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='y=x (perfect match)')

    ax.set_xlabel('Validation Targets', color='black', fontweight='bold')
    ax.set_ylabel('Synthetic Targets', color='black', fontweight='bold')
    ax.set_title('Target Values: Validation vs Synthetic', color='black', fontweight='bold', pad=15)
    ax.legend()
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_volatility_clustering(
    acf_abs_val: np.ndarray,
    acf_abs_syn: np.ndarray,
    max_lag: int,
    save_path: Optional[Path] = None
):
    """
    Plot volatility clustering comparison (ACF of absolute values).

    Args:
        acf_abs_val: ACF of absolute validation targets
        acf_abs_syn: ACF of absolute synthetic targets
        max_lag: Maximum lag
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lags = np.arange(0, max_lag + 1)

    ax.plot(lags, acf_abs_val, 'o-', color=COLORS['non_significant'],
            linewidth=2, markersize=6, label='Validation |target|', alpha=0.8)
    ax.plot(lags, acf_abs_syn, 's-', color=COLORS['significant'],
            linewidth=2, markersize=6, label='Synthetic |target|', alpha=0.8)

    # Add 95% confidence interval
    n_samples = 1000
    ci = 1.96 / np.sqrt(n_samples)
    ax.axhline(y=ci, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-ci, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Lag', color='black', fontweight='bold')
    ax.set_ylabel('Autocorrelation of |Target|', color='black', fontweight='bold')
    ax.set_title('Volatility Clustering (GARCH Effect)', color='black', fontweight='bold', pad=15)
    ax.legend()
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
