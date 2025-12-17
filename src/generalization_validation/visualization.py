"""Visualization utilities for generalization validation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional, Tuple


def plot_umap_comparison(
    X_original: np.ndarray,
    X_synthetic: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    method: str = 'umap',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: int = 30
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

    Returns:
        Tuple of (original_embedding, synthetic_embedding)
    """
    # Combine data
    X_combined = np.vstack([X_original, X_synthetic])
    labels = np.array(['Original'] * len(X_original) + ['Synthetic'] * len(X_synthetic))

    # Dimensionality reduction
    if method == 'umap':
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
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
        c='blue', alpha=0.3, s=10, label='Original', rasterized=True
    )
    scatter_syn = ax.scatter(
        embedding_syn[:, 0], embedding_syn[:, 1],
        c='red', alpha=0.3, s=10, label='Synthetic', rasterized=True
    )

    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    axes[0].bar(range(len(reconstruction_errors)), reconstruction_errors, width=1.0)
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Per-Feature Reconstruction Error')
    axes[0].grid(True, alpha=0.3)

    # Histogram of errors
    axes[1].hist(reconstruction_errors, bins=bins, edgecolor='black')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Reconstruction Errors')
    axes[1].grid(True, alpha=0.3)

    # Add statistics
    mean_err = np.mean(reconstruction_errors)
    median_err = np.median(reconstruction_errors)
    max_err = np.max(reconstruction_errors)
    axes[1].axvline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.4f}')
    axes[1].axvline(median_err, color='green', linestyle='--', label=f'Median: {median_err:.4f}')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

    axes[0].bar(x - width/2, freq_val, width, label='Validation', alpha=0.7)
    axes[0].bar(x + width/2, freq_syn, width, label='Synthetic', alpha=0.7)
    axes[0].set_xlabel('Code Index')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Code Frequency Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot for correlation
    axes[1].scatter(freq_val, freq_syn, alpha=0.5, s=30)
    axes[1].plot([0, freq_val.max()], [0, freq_val.max()], 'r--', label='y=x')
    axes[1].set_xlabel('Validation Frequency')
    axes[1].set_ylabel('Synthetic Frequency')
    axes[1].set_title('Code Frequency Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Compute correlation
    corr = np.corrcoef(freq_val, freq_syn)[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr:.4f}',
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    sns.heatmap(trans_val, ax=axes[0], cmap='viridis', vmin=0, vmax=vmax, cbar=True)
    axes[0].set_title('Validation Transition Matrix')
    axes[0].set_xlabel('To Code')
    axes[0].set_ylabel('From Code')

    # Synthetic transition matrix
    sns.heatmap(trans_syn, ax=axes[1], cmap='viridis', vmin=0, vmax=vmax, cbar=True)
    axes[1].set_title('Synthetic Transition Matrix')
    axes[1].set_xlabel('To Code')
    axes[1].set_ylabel('From Code')

    # Difference matrix
    diff = np.abs(trans_val - trans_syn)
    sns.heatmap(diff, ax=axes[2], cmap='Reds', vmin=0, cbar=True)
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('To Code')
    axes[2].set_ylabel('From Code')

    # Add Frobenius norm
    frobenius = np.linalg.norm(trans_val - trans_syn, ord='fro')
    axes[2].text(0.5, -0.15, f'Frobenius Norm: {frobenius:.4f}',
                transform=axes[2].transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Validation
    ngrams_val = [str(ng) for ng, _ in top_val[:k]]
    counts_val = [cnt for _, cnt in top_val[:k]]

    axes[0].barh(range(k), counts_val, tick_label=ngrams_val)
    axes[0].set_xlabel('Count')
    axes[0].set_title(f'Top-{k} Validation {n}-grams')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)

    # Synthetic
    ngrams_syn = [str(ng) for ng, _ in top_syn[:k]]
    counts_syn = [cnt for _, cnt in top_syn[:k]]

    axes[1].barh(range(k), counts_syn, tick_label=ngrams_syn)
    axes[1].set_xlabel('Count')
    axes[1].set_title(f'Top-{k} Synthetic {n}-grams')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    sns.heatmap(corr_orig_vis, ax=axes[0], cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, cbar=True)
    axes[0].set_title('Original Correlation')

    # Synthetic
    sns.heatmap(corr_syn_vis, ax=axes[1], cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, cbar=True)
    axes[1].set_title('Synthetic Correlation')

    # Difference
    diff = np.abs(corr_orig_vis - corr_syn_vis)
    sns.heatmap(diff, ax=axes[2], cmap='Reds', vmin=0, square=True, cbar=True)
    axes[2].set_title('Absolute Difference')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
