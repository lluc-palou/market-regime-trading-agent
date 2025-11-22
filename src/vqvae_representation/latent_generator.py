"""
Latent Generator

Generates and materializes latent representations (codebook indices) for all samples.

Strategy:
- GPU efficiency: Process samples in hour groups (100 hours) with mini-batches (2048)
- Memory safety: Write to MongoDB hour-by-hour (~120 documents per insert)
- Preserves all original document fields except 'bins' vector
"""

import torch
from datetime import datetime, timedelta
from typing import List, Dict
from pymongo import MongoClient

from src.utils.logging import logger
from .data_loader import load_hourly_batch
from .config import TRAINING_CONFIG


class LatentGenerator:
    """
    Generates latent codes and materializes them to MongoDB.
    
    Uses hour accumulation for GPU efficiency while writing
    hour-by-hour for memory safety.
    """
    
    def __init__(
        self,
        spark,
        db_name: str,
        split_collection: str,
        output_collection: str,
        model: torch.nn.Module,
        device: torch.device,
        mongo_uri: str = "mongodb://127.0.0.1:27017/"
    ):
        """
        Initialize latent generator.
        
        Args:
            spark: SparkSession instance
            db_name: Database name
            split_collection: Input collection name (e.g., 'split_0_input')
            output_collection: Output collection name (e.g., 'split_0_output')
            model: Trained VQ-VAE model (frozen)
            device: torch device
            mongo_uri: MongoDB connection URI
        """
        self.spark = spark
        self.db_name = db_name
        self.split_collection = split_collection
        self.output_collection = output_collection
        self.model = model
        self.device = device
        
        # MongoDB connection for writing
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.output_col = self.db[output_collection]
        
        # Set model to eval mode
        self.model.eval()
    
    def generate_and_write_latents(
        self,
        all_hours: List[datetime],
        split_id: int
    ) -> Dict:
        """
        Generate latent codes for all samples and write to output collection.
        
        Process:
        1. Accumulate hours (100 at a time) for GPU efficiency
        2. Process in mini-batches (2048) on GPU
        3. Write to MongoDB hour-by-hour for memory safety
        
        Args:
            all_hours: List of hourly time windows
            split_id: Split identifier
            
        Returns:
            Dictionary with generation statistics
        """
        logger('', "INFO")
        logger('Generating latent representations...', "INFO")
        logger('=' * 80, "INFO")
        
        hours_per_acc = TRAINING_CONFIG['hours_per_accumulation']
        mini_batch_size = TRAINING_CONFIG['mini_batch_size']
        
        total_samples_processed = 0
        total_train_samples = 0
        total_val_samples = 0
        
        # Process hours in groups
        num_hour_groups = (len(all_hours) + hours_per_acc - 1) // hours_per_acc
        
        for group_idx in range(0, len(all_hours), hours_per_acc):
            hour_group = all_hours[group_idx:min(group_idx + hours_per_acc, len(all_hours))]
            current_group = group_idx // hours_per_acc + 1
            
            logger(
                f'Processing hour group {current_group}/{num_hour_groups} '
                f'({len(hour_group)} hours)',
                "INFO"
            )
            
            # Generate latents for this hour group
            stats = self._process_hour_group(hour_group, split_id, mini_batch_size)
            
            total_samples_processed += stats['samples_processed']
            total_train_samples += stats['train_samples']
            total_val_samples += stats['val_samples']
            
            logger(
                f'  Processed {stats["samples_processed"]:,} samples '
                f'({stats["train_samples"]:,} train, {stats["val_samples"]:,} val)',
                "INFO"
            )
        
        logger('', "INFO")
        logger(f'Latent generation complete:', "INFO")
        logger(f'  Total samples: {total_samples_processed:,}', "INFO")
        logger(f'  Training samples: {total_train_samples:,}', "INFO")
        logger(f'  Validation samples: {total_val_samples:,}', "INFO")
        logger(f'  Output collection: {self.output_collection}', "INFO")
        
        return {
            'total_samples': total_samples_processed,
            'train_samples': total_train_samples,
            'val_samples': total_val_samples
        }
    
    def _process_hour_group(
        self,
        hour_group: List[datetime],
        split_id: int,
        mini_batch_size: int
    ) -> Dict:
        """
        Process one group of hours (typically 100).
        
        Args:
            hour_group: List of hours to process
            split_id: Split identifier
            mini_batch_size: Mini-batch size for GPU processing
            
        Returns:
            Statistics dictionary
        """
        # Load all samples for this hour group (both train and validation)
        hour_data = []  # List of (hour, role, samples_tensor, original_docs)
        
        for hour in hour_group:
            hour_end = hour + timedelta(hours=1)
            
            # Load training samples
            train_batch = load_hourly_batch(
                self.spark,
                self.db_name,
                self.split_collection,
                hour,
                hour_end,
                role='train'
            )
            
            if train_batch is not None:
                train_docs = self._fetch_original_documents(hour, hour_end, 'train')
                hour_data.append(('train', train_batch, train_docs))
            
            # Load validation samples
            val_batch = load_hourly_batch(
                self.spark,
                self.db_name,
                self.split_collection,
                hour,
                hour_end,
                role='validation'
            )
            
            if val_batch is not None:
                val_docs = self._fetch_original_documents(hour, hour_end, 'validation')
                hour_data.append(('validation', val_batch, val_docs))
        
        # Process all accumulated samples through model
        total_processed = 0
        train_count = 0
        val_count = 0
        
        for role, samples_tensor, original_docs in hour_data:
            # Generate latent codes in mini-batches
            latent_indices = []
            
            num_samples = samples_tensor.size(0)
            
            with torch.no_grad():
                for i in range(0, num_samples, mini_batch_size):
                    mini_batch = samples_tensor[i:i+mini_batch_size].to(self.device)
                    
                    # Get codebook indices from model
                    indices = self.model.encode(mini_batch)
                    latent_indices.append(indices.cpu())
            
            # Combine all indices
            latent_indices = torch.cat(latent_indices, dim=0).numpy()
            
            # Write documents to output collection
            self._write_documents_with_latents(
                original_docs,
                latent_indices,
                split_id
            )
            
            total_processed += num_samples
            if role == 'train':
                train_count += num_samples
            else:
                val_count += num_samples
        
        return {
            'samples_processed': total_processed,
            'train_samples': train_count,
            'val_samples': val_count
        }
    
    def _fetch_original_documents(
        self,
        hour_start: datetime,
        hour_end: datetime,
        role: str
    ) -> List[Dict]:
        """
        Fetch original documents from MongoDB.
        
        Args:
            hour_start: Start of hour window
            hour_end: End of hour window
            role: 'train' or 'validation'
            
        Returns:
            List of original documents
        """
        # Read from Spark for consistency with data loading
        df = self.spark.read \
            .format("mongodb") \
            .option("database", self.db_name) \
            .option("collection", self.split_collection) \
            .load()
        
        # Filter by hour and role
        df = df.filter(
            (df.timestamp >= hour_start) &
            (df.timestamp < hour_end) &
            (df.role == role)
        )
        
        # Convert to list of dicts
        docs = df.toPandas().to_dict('records')
        
        return docs
    
    def _write_documents_with_latents(
        self,
        original_docs: List[Dict],
        latent_indices: list,
        split_id: int
    ):
        """
        Write documents with latent codes to output collection.

        Preserves all original fields INCLUDING 'bins' vector.
        Adds 'codebook_index' field.

        Args:
            original_docs: Original documents from input collection
            latent_indices: Codebook indices (numpy array)
            split_id: Split identifier
        """
        if len(original_docs) != len(latent_indices):
            raise ValueError(
                f"Mismatch: {len(original_docs)} documents but {len(latent_indices)} latent codes"
            )

        # Prepare documents for insertion
        output_docs = []

        for doc, latent_idx in zip(original_docs, latent_indices):
            # Copy all fields except '_id' (KEEP bins array)
            output_doc = {
                k: v for k, v in doc.items()
                if k != '_id'
            }

            # Add latent code
            output_doc['codebook_index'] = int(latent_idx)

            # Ensure split_id is consistent
            output_doc['split_id'] = split_id

            output_docs.append(output_doc)

        # Batch insert to MongoDB
        if output_docs:
            self.output_col.insert_many(output_docs, ordered=False)
    
    def close(self):
        """Close MongoDB connection."""
        self.mongo_client.close()