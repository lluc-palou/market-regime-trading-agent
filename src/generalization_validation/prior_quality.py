"""Experiment 2: Prior Model Quality Assessment."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.logging import logger
from .data_loader import load_validation_samples, load_synthetic_samples, organize_codes_into_sequences
from .metrics import (
    compute_code_frequency,
    compute_transition_matrix,
    extract_ngrams,
    compare_ngrams,
    jensen_shannon_divergence
)
from .visualization import (
    plot_code_frequency,
    plot_transition_matrix,
    plot_ngram_comparison,
    plot_umap_comparison
)


class PriorQualityValidator:
    """Validates Prior model's ability to generate realistic code sequences."""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        output_dir: Path,
        device: torch.device,
        seq_len: int = 120
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.output_dir = output_dir
        self.device = device
        self.seq_len = seq_len

        # Create output directory
        (self.output_dir / "experiment2_prior_quality").mkdir(parents=True, exist_ok=True)

    def validate_split(self, split_id: int, vocab_size: int = 128) -> Dict:
        """
        Run Prior quality validation for one split using pre-generated synthetic data.

        Args:
            split_id: Split identifier
            vocab_size: Vocabulary size (default: 128)

        Returns:
            Dictionary with validation metrics
        """
        logger('', "INFO")
        logger(f'Validating Prior model for split {split_id}...', "INFO")

        # Load validation codes
        _, val_codebook_indices, _ = load_validation_samples(
            self.mongo_uri, self.db_name, split_id
        )

        logger(f'  Validation codes: {len(val_codebook_indices):,}', "INFO")

        # Organize validation codes into sequences
        val_sequences = organize_codes_into_sequences(
            val_codebook_indices, self.seq_len, stride=self.seq_len
        )

        n_val_sequences = len(val_sequences)
        logger(f'  Validation sequences: {n_val_sequences}', "INFO")

        # Load pre-generated synthetic data
        _, syn_codebook_indices, syn_sequence_ids = load_synthetic_samples(
            self.mongo_uri, self.db_name, split_id
        )

        logger(f'  Synthetic codes: {len(syn_codebook_indices):,}', "INFO")

        # Organize synthetic codes into sequences using sequence_id
        # Each unique sequence_id represents one sequence of length seq_len
        unique_seq_ids = np.unique(syn_sequence_ids)
        syn_sequences = []

        for seq_id in unique_seq_ids:
            mask = syn_sequence_ids == seq_id
            seq_codes = syn_codebook_indices[mask]

            # Ensure correct length (should be seq_len)
            if len(seq_codes) == self.seq_len:
                syn_sequences.append(seq_codes)
            else:
                logger(f'  Warning: Sequence {seq_id} has length {len(seq_codes)}, expected {self.seq_len}', "WARNING")

        syn_sequences = np.array(syn_sequences, dtype=np.int64)
        n_syn_sequences = len(syn_sequences)
        logger(f'  Synthetic sequences: {n_syn_sequences}', "INFO")

        # 1. Code frequency distributions
        logger('  Analyzing code frequencies...', "INFO")
        freq_val = compute_code_frequency(val_sequences, vocab_size)
        freq_syn = compute_code_frequency(syn_sequences, vocab_size)

        # JS divergence
        js_div = jensen_shannon_divergence(freq_val, freq_syn)
        logger(f'  JS Divergence (frequencies): {js_div:.6f}', "INFO")

        # Frequency correlation (suppress warning for zero-variance features)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in divide')
            freq_corr = np.corrcoef(freq_val, freq_syn)[0, 1]
        freq_corr = 0.0 if np.isnan(freq_corr) else freq_corr
        logger(f'  Frequency correlation: {freq_corr:.6f}', "INFO")

        # 2. Transition probabilities
        logger('  Computing transition matrices...', "INFO")
        trans_val = compute_transition_matrix(val_sequences, vocab_size)
        trans_syn = compute_transition_matrix(syn_sequences, vocab_size)

        # Frobenius norm (unbounded, for reference)
        trans_frobenius_norm = np.linalg.norm(trans_val - trans_syn, ord='fro')

        # Frobenius correlation (bounded [-1, 1], more interpretable)
        norm_val = np.linalg.norm(trans_val, ord='fro')
        norm_syn = np.linalg.norm(trans_syn, ord='fro')
        if norm_val > 0 and norm_syn > 0:
            trans_frobenius_corr = np.trace(trans_val.T @ trans_syn) / (norm_val * norm_syn)
        else:
            trans_frobenius_corr = 0.0

        # Mean absolute difference (bounded [0, 1] for probability matrices)
        trans_mad = np.mean(np.abs(trans_val - trans_syn))

        logger(f'  Transition matrix Frobenius correlation: {trans_frobenius_corr:.6f}', "INFO")
        logger(f'  Transition matrix mean absolute diff: {trans_mad:.6f}', "INFO")

        # 3. N-gram statistics
        logger('  Extracting n-grams...', "INFO")

        # Bigrams
        bigrams_val = extract_ngrams(val_sequences, 2)
        bigrams_syn = extract_ngrams(syn_sequences, 2)
        bigram_comparison = compare_ngrams(bigrams_val, bigrams_syn, top_k=20)

        logger(f'  Bigram overlap ratio: {bigram_comparison["overlap_ratio"]:.4f}', "INFO")
        logger(f'  Bigram frequency correlation: {bigram_comparison["frequency_correlation"]:.6f}', "INFO")

        # Trigrams
        trigrams_val = extract_ngrams(val_sequences, 3)
        trigrams_syn = extract_ngrams(syn_sequences, 3)
        trigram_comparison = compare_ngrams(trigrams_val, trigrams_syn, top_k=20)

        logger(f'  Trigram overlap ratio: {trigram_comparison["overlap_ratio"]:.4f}', "INFO")

        # Visualizations
        split_output_dir = self.output_dir / "experiment2_prior_quality" / f"split_{split_id}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Code frequency plot
        logger('  Plotting code frequencies...', "INFO")
        plot_code_frequency(
            freq_val, freq_syn,
            save_path=split_output_dir / f"code_frequency_split_{split_id}.png"
        )

        # Transition matrix plot
        logger('  Plotting transition matrices...', "INFO")
        plot_transition_matrix(
            trans_val, trans_syn,
            save_path=split_output_dir / f"transition_matrix_split_{split_id}.png"
        )

        # Bigram comparison plot
        logger('  Plotting bigrams...', "INFO")
        plot_ngram_comparison(
            bigram_comparison['top_val_ngrams'],
            bigram_comparison['top_syn_ngrams'],
            n=2,
            save_path=split_output_dir / f"bigrams_split_{split_id}.png"
        )

        # Trigram comparison plot
        logger('  Plotting trigrams...', "INFO")
        plot_ngram_comparison(
            trigram_comparison['top_val_ngrams'],
            trigram_comparison['top_syn_ngrams'],
            n=3,
            save_path=split_output_dir / f"trigrams_split_{split_id}.png"
        )

        # UMAP on sequences (treat sequences as high-dimensional points)
        logger('  Generating sequence UMAP...', "INFO")
        plot_umap_comparison(
            val_sequences.astype(np.float32),
            syn_sequences.astype(np.float32),
            title=f'Prior Sequences - Split {split_id}',
            save_path=split_output_dir / f"umap_sequences_split_{split_id}.png",
            method='umap'
        )

        # Compile results
        results = {
            'split_id': split_id,
            'n_val_sequences': n_val_sequences,
            'n_syn_sequences': n_syn_sequences,
            'seq_len': self.seq_len,
            'vocab_size': vocab_size,
            'js_divergence_freq': float(js_div),
            'frequency_correlation': float(freq_corr),
            'transition_frobenius_correlation': float(trans_frobenius_corr),
            'transition_mean_abs_diff': float(trans_mad),
            'bigram_overlap_ratio': float(bigram_comparison['overlap_ratio']),
            'bigram_freq_correlation': float(bigram_comparison['frequency_correlation']),
            'trigram_overlap_ratio': float(trigram_comparison['overlap_ratio']),
            'trigram_freq_correlation': float(trigram_comparison['frequency_correlation'])
        }

        logger(f'  ✓ Split {split_id} validation complete', "INFO")

        return results
