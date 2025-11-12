"""
Prior Data Loader

Handles loading and sequencing of latent codes for prior model training.
"""

import torch
from datetime import datetime, timedelta
from typing import List, Tuple
from pyspark.sql import SparkSession

from src.utils.logging import logger


def load_latent_codes_for_hours(
    spark: SparkSession,
    db_name: str,
    collection: str,
    hour_start: datetime,
    hour_end: datetime,
    role: str = None
) -> List[Tuple[datetime, int]]:
    """
    Load latent codes for a time range.
    
    Args:
        spark: SparkSession
        db_name: Database name
        collection: Collection name (e.g., 'split_0_input')
        hour_start: Start time
        hour_end: End time
        role: Optional role filter ('train' or 'validation')
        
    Returns:
        List of (timestamp, codebook_index) tuples
    """
    df = spark.read \
        .format("mongodb") \
        .option("database", db_name) \
        .option("collection", collection) \
        .load()
    
    # Filter by time range
    df = df.filter(
        (df.timestamp >= hour_start) &
        (df.timestamp < hour_end)
    )
    
    # Filter by role if specified
    if role is not None:
        df = df.filter(df.role == role)
    
    # Select only timestamp and codebook_index
    df = df.select('timestamp', 'codebook_index').orderBy('timestamp')
    
    # Collect to list
    rows = df.collect()
    
    if not rows:
        return []
    
    return [(row.timestamp, row.codebook_index) for row in rows]


def create_sequences_from_codes(
    latent_codes: List[Tuple[datetime, int]],
    seq_len: int = 120
) -> Tuple[torch.Tensor, List[datetime]]:
    """
    Create non-overlapping sequences from latent codes.
    
    Args:
        latent_codes: List of (timestamp, code_index) tuples (sorted by time)
        seq_len: Sequence length
        
    Returns:
        sequences: (num_sequences, seq_len) tensor of codes
        start_times: List of start timestamps for each sequence
    """
    if len(latent_codes) < seq_len:
        return None, None
    
    # Extract codes (already sorted by timestamp)
    codes = [code for _, code in latent_codes]
    timestamps = [ts for ts, _ in latent_codes]
    
    # Create non-overlapping sequences
    sequences = []
    start_times = []
    
    for i in range(0, len(codes) - seq_len + 1, seq_len):
        seq = codes[i:i+seq_len]
        if len(seq) == seq_len:
            sequences.append(seq)
            start_times.append(timestamps[i])
    
    if not sequences:
        return None, None
    
    # Convert to tensor
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    
    return sequences_tensor, start_times


def load_sequences_for_split(
    spark: SparkSession,
    db_name: str,
    collection: str,
    all_hours: List[datetime],
    role: str,
    seq_len: int = 120,
    hours_per_accumulation: int = 100
) -> List[torch.Tensor]:
    """
    Load sequences for a split using hour accumulation.
    
    Args:
        spark: SparkSession
        db_name: Database name
        collection: Collection name
        all_hours: List of hour timestamps
        role: 'train' or 'validation'
        seq_len: Sequence length
        hours_per_accumulation: Hours to load at once
        
    Returns:
        List of sequence tensors (each tensor has multiple sequences)
    """
    all_sequences = []
    
    # Process hours in groups
    for hour_idx in range(0, len(all_hours), hours_per_accumulation):
        hour_group = all_hours[hour_idx:min(hour_idx + hours_per_accumulation, len(all_hours))]
        
        if not hour_group:
            continue
        
        # Load latent codes for this hour group
        hour_start = hour_group[0]
        hour_end = hour_group[-1] + timedelta(hours=1)
        
        latent_codes = load_latent_codes_for_hours(
            spark, db_name, collection,
            hour_start, hour_end, role
        )
        
        if not latent_codes:
            continue
        
        # Create sequences
        sequences, _ = create_sequences_from_codes(latent_codes, seq_len)
        
        if sequences is not None:
            all_sequences.append(sequences)
    
    return all_sequences


def get_sequence_dataset_size(
    spark: SparkSession,
    db_name: str,
    collection: str,
    role: str,
    seq_len: int = 120
) -> int:
    """
    Estimate number of sequences for a role.
    
    Args:
        spark: SparkSession
        db_name: Database name
        collection: Collection name
        role: 'train' or 'validation'
        seq_len: Sequence length
        
    Returns:
        Estimated number of sequences
    """
    df = spark.read \
        .format("mongodb") \
        .option("database", db_name) \
        .option("collection", collection) \
        .load()
    
    count = df.filter(df.role == role).count()
    
    # Estimate number of sequences (non-overlapping)
    num_sequences = count // seq_len
    
    return num_sequences