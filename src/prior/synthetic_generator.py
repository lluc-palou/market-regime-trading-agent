"""
Synthetic LOB Generator

Generates synthetic LOB sequences using trained prior and VQ-VAE decoder.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pymongo import MongoClient

from src.utils.logging import logger
from .prior_model import LatentPriorCNN
from .prior_config import GENERATION_CONFIG


class SyntheticLOBGenerator:
    """
    Generates synthetic LOB sequences.
    """
    
    def __init__(
        self,
        prior_model: LatentPriorCNN,
        vqvae_model,
        device: torch.device,
        mongo_uri: str = "mongodb://127.0.0.1:27017/"
    ):
        """
        Initialize generator.
        
        Args:
            prior_model: Trained prior model
            vqvae_model: Trained VQ-VAE model
            device: torch device
            mongo_uri: MongoDB connection URI
        """
        self.prior = prior_model.eval()
        self.vqvae = vqvae_model.eval()
        self.device = device
        
        # MongoDB connection
        self.mongo_client = MongoClient(mongo_uri)
    
    def generate_and_save(
        self,
        db_name: str,
        output_collection: str,
        split_id: int,
        n_sequences: int = None,
        seq_len: int = None,
        temperature: float = None
    ) -> Dict:
        """
        Generate synthetic sequences and save to MongoDB.
        
        Args:
            db_name: Database name
            output_collection: Output collection name
            split_id: Split identifier
            n_sequences: Number of sequences (default from config)
            seq_len: Sequence length (default from config)
            temperature: Sampling temperature (default from config)
            
        Returns:
            Generation statistics
        """
        if n_sequences is None:
            n_sequences = GENERATION_CONFIG['sequences_per_split']
        if seq_len is None:
            seq_len = GENERATION_CONFIG['seq_len']
        if temperature is None:
            temperature = GENERATION_CONFIG['temperature']
        
        logger('', "INFO")
        logger(f'Generating {n_sequences} synthetic sequences...', "INFO")
        logger(f'Sequence length: {seq_len}, Temperature: {temperature}', "INFO")
        
        db = self.mongo_client[db_name]
        output_col = db[output_collection]
        
        # Clear collection if exists
        if output_collection in db.list_collection_names():
            output_col.drop()
        
        total_samples = 0
        batch_size = GENERATION_CONFIG['generation_batch_size']
        
        # Generate in batches
        for batch_idx in range(0, n_sequences, batch_size):
            batch_n = min(batch_size, n_sequences - batch_idx)
            
            # Sample latent codes from prior
            with torch.no_grad():
                z_batch = self.prior.sample(
                    n_samples=batch_n,
                    seq_len=seq_len,
                    temperature=temperature,
                    device=self.device
                )  # (batch_n, seq_len)
            
            # Decode each sequence
            for seq_idx in range(batch_n):
                sequence_id = batch_idx + seq_idx
                
                # Decode sequence
                lob_sequence = self._decode_sequence(z_batch[seq_idx])
                
                # Save to MongoDB
                self._save_sequence_to_db(
                    output_col,
                    lob_sequence,
                    z_batch[seq_idx].cpu().numpy(),
                    split_id,
                    sequence_id
                )
                
                total_samples += seq_len
            
            if (batch_idx + batch_n) % 100 == 0 or (batch_idx + batch_n) == n_sequences:
                logger(f'Generated {batch_idx + batch_n}/{n_sequences} sequences', "INFO")
        
        logger(f'Generation complete: {total_samples} samples in {n_sequences} sequences', "INFO")
        
        return {
            'total_sequences': n_sequences,
            'total_samples': total_samples,
            'seq_len': seq_len,
            'temperature': temperature
        }
    
    def _decode_sequence(self, z_seq: torch.Tensor) -> np.ndarray:
        """
        Decode a sequence of latent codes to LOB vectors.
        
        Args:
            z_seq: (seq_len,) tensor of discrete codes
            
        Returns:
            lob_sequence: (seq_len, B) array of LOB vectors
        """
        seq_len = z_seq.size(0)
        lob_vectors = []
        
        with torch.no_grad():
            for t in range(seq_len):
                code_idx = z_seq[t].item()
                
                # Get code embedding from VQ-VAE codebook
                code_embedding = self.vqvae.vq.embedding.weight[code_idx].unsqueeze(0)
                
                # Decode to LOB vector
                lob_vector = self.vqvae.decoder(code_embedding)
                
                lob_vectors.append(lob_vector.cpu().numpy()[0])
        
        return np.array(lob_vectors)
    
    def _save_sequence_to_db(
        self,
        collection,
        lob_sequence: np.ndarray,
        latent_codes: np.ndarray,
        split_id: int,
        sequence_id: int
    ):
        """
        Save synthetic sequence to MongoDB.
        
        Args:
            collection: MongoDB collection
            lob_sequence: (seq_len, B) LOB vectors
            latent_codes: (seq_len,) discrete codes
            split_id: Split identifier
            sequence_id: Sequence identifier
        """
        seq_len = lob_sequence.shape[0]
        
        # Create synthetic timestamps (arbitrary start)
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        documents = []
        for t in range(seq_len):
            doc = {
                'timestamp': base_time + timedelta(seconds=30 * (sequence_id * seq_len + t)),
                'codebook_index': int(latent_codes[t]),
                'bins': lob_sequence[t].tolist(),
                'split_id': split_id,
                'sequence_id': sequence_id,
                'position_in_sequence': t,
                'is_synthetic': True,
                'generation_metadata': {
                    'temperature': GENERATION_CONFIG['temperature'],
                    'generation_date': datetime.now().isoformat()
                }
            }
            documents.append(doc)
        
        # Batch insert
        if documents:
            collection.insert_many(documents, ordered=False)
    
    def close(self):
        """Close MongoDB connection."""
        self.mongo_client.close()


def load_models_for_generation(
    prior_model_path: str,
    vqvae_model_path: str,
    device: torch.device
) -> Tuple[LatentPriorCNN, object]:
    """
    Load trained prior and VQ-VAE models.
    
    Args:
        prior_model_path: Path to prior model checkpoint
        vqvae_model_path: Path to VQ-VAE model checkpoint
        device: torch device
        
    Returns:
        prior_model: Loaded prior model
        vqvae_model: Loaded VQ-VAE model
    """
    # Load prior
    prior_checkpoint = torch.load(prior_model_path, map_location=device)
    
    from .prior_model import LatentPriorCNN
    prior_model = LatentPriorCNN(
        codebook_size=prior_checkpoint['codebook_size'],
        embedding_dim=prior_checkpoint['config']['embedding_dim'],
        n_layers=prior_checkpoint['config']['n_layers'],
        n_channels=prior_checkpoint['config']['n_channels'],
        kernel_size=prior_checkpoint['config']['kernel_size'],
        dropout=prior_checkpoint['config']['dropout']
    ).to(device)
    
    prior_model.load_state_dict(prior_checkpoint['model_state_dict'])
    prior_model.eval()
    
    # Load VQ-VAE
    vqvae_checkpoint = torch.load(vqvae_model_path, map_location=device)
    
    from src.vqvae_representation.model import VQVAEModel
    vqvae_model = VQVAEModel(vqvae_checkpoint['config']).to(device)
    vqvae_model.load_state_dict(vqvae_checkpoint['model_state_dict'])
    vqvae_model.eval()
    
    return prior_model, vqvae_model