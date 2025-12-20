"""
VQ-VAE Model Architecture

Components:
- VectorQuantizer: Codebook with straight-through estimator
- Encoder: Convolutional encoder with dropout regularization
- Decoder: Transposed convolutional decoder with dropout regularization
- VQVAEModel: Complete model wrapper
- Loss functions: MSE and Wasserstein distance for distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .config import MODEL_CONFIG


def wasserstein_1d_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute normalized 1-Wasserstein distance (Earth Mover's Distance) for 1D distributions.

    For discrete distributions over ordered bins, the 1-Wasserstein distance is:
    W1(p, q) = sum_i |CDF_p(i) - CDF_q(i)|

    This is superior to MSE for probability distributions because it considers
    the geometry of the space - moving mass between nearby bins costs less than
    moving it between distant bins.

    The result is normalized by the number of bins to be comparable to MSE loss
    (both in 0-1 range), ensuring hyperparameters (beta, usage_penalty) remain balanced.

    Args:
        pred: Predicted distribution (batch_size, num_bins)
        target: Target distribution (batch_size, num_bins)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Wasserstein distance loss (normalized to 0-1 range like MSE)
    """
    # Ensure non-negative values (distributions should be non-negative)
    pred = torch.clamp(pred, min=0.0)
    target = torch.clamp(target, min=0.0)

    # Normalize to ensure they sum to the same value (distribution constraint)
    # This handles cases where reconstruction doesn't perfectly preserve mass
    pred_sum = pred.sum(dim=1, keepdim=True).clamp(min=1e-8)
    target_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    pred_normalized = pred / pred_sum
    target_normalized = target / target_sum

    # Compute CDFs (cumulative distribution functions)
    pred_cdf = torch.cumsum(pred_normalized, dim=1)
    target_cdf = torch.cumsum(target_normalized, dim=1)

    # 1-Wasserstein distance = L1 distance between CDFs
    # CRITICAL: Normalize by number of bins to make comparable to MSE (both in 0-1 range)
    # Without normalization, loss scales with num_bins (1001), making it ~100-1000x larger than MSE
    num_bins = pred.shape[1]
    wasserstein_dist = torch.abs(pred_cdf - target_cdf).sum(dim=1) / num_bins

    if reduction == 'mean':
        return wasserstein_dist.mean()
    elif reduction == 'sum':
        return wasserstein_dist.sum()
    elif reduction == 'none':
        return wasserstein_dist
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean', 'sum', or 'none'.")


class VectorQuantizer(nn.Module):
    """
    Vector quantization layer with learnable codebook.

    Supports both gradient-based and EMA-based codebook updates.
    EMA updates improve training stability and prevent codebook collapse.

    Uses straight-through estimator for gradient flow during backprop.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 use_ema: bool = False, ema_decay: float = 0.99, ema_epsilon: float = 1e-5):
        """
        Initialize vector quantizer.

        Args:
            num_embeddings: Number of codebook vectors (K)
            embedding_dim: Dimension of each codebook vector (D)
            use_ema: Use EMA updates instead of gradient-based updates
            ema_decay: EMA decay rate (gamma), typically 0.99
            ema_epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        # Learnable codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Xavier initialization for better initial diversity and prevents early collapse
        # Old: uniform(-1/K, 1/K) gave tiny range (e.g., -0.008 to 0.008 for K=128)
        # Xavier: scales based on both K and D for optimal variance
        nn.init.xavier_uniform_(self.embedding.weight)

        if use_ema:
            # EMA cluster size (N_i): running count of samples assigned to each code
            # Initialize to 1.0 (not 0) to prevent division by near-zero for unused codes
            self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
            # EMA embedding average (m_i): running average of encoder outputs per code
            self.register_buffer('ema_embedding_avg', self.embedding.weight.data.clone())
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize encoder outputs to nearest codebook vectors.

        Args:
            z_e: Encoder output (batch_size, embedding_dim)

        Returns:
            z_q: Quantized vectors (batch_size, embedding_dim)
            commitment_loss: Commitment loss for encoder
            codebook_loss: Codebook loss for embeddings (0 if using EMA)
            encoding_indices: Indices of selected codebook vectors (batch_size,)
        """
        B, D = z_e.shape

        # Compute distances to all codebook vectors
        # ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2*z_e·e
        distances = (
            torch.sum(z_e**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.matmul(z_e, self.embedding.weight.t())
        )

        # Find nearest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)

        # Compute commitment loss (always used)
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        if self.use_ema and self.training:
            # EMA codebook updates (no gradient-based loss)
            self._ema_update(z_e, encoding_indices)
            codebook_loss = torch.tensor(0.0, device=z_e.device)
        else:
            # Gradient-based codebook update
            codebook_loss = F.mse_loss(z_e.detach(), z_q)

        # Straight-through estimator: copy gradients from decoder to encoder
        z_q = z_e + (z_q - z_e).detach()

        return z_q, commitment_loss, codebook_loss, encoding_indices

    def _ema_update(self, z_e: torch.Tensor, encoding_indices: torch.Tensor):
        """
        Update codebook using EMA.

        Updates cluster counts and embedding averages, then recomputes embeddings.

        Args:
            z_e: Encoder outputs (batch_size, embedding_dim)
            encoding_indices: Selected code indices (batch_size,)
        """
        # Create one-hot encodings: (batch_size, num_embeddings)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Update cluster sizes with EMA
        # N_i = γ * N_i + (1 - γ) * n_i
        # n_i = number of samples assigned to code i in current batch
        n_i = encodings.sum(0)  # (num_embeddings,)
        self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + n_i * (1 - self.ema_decay)

        # Update embedding averages with EMA
        # m_i = γ * m_i + (1 - γ) * sum(z_e for samples assigned to i)
        dw = torch.matmul(encodings.t(), z_e)  # (num_embeddings, embedding_dim)
        self.ema_embedding_avg = self.ema_embedding_avg * self.ema_decay + dw * (1 - self.ema_decay)

        # Normalize to get updated embeddings
        # e_i = m_i / N_i
        # Add epsilon for numerical stability (Laplace smoothing)
        cluster_size = self.ema_cluster_size + self.ema_epsilon

        # Update embedding weights (no gradient needed since we're in training mode)
        self.embedding.weight.data = self.ema_embedding_avg / cluster_size.unsqueeze(1)
    
    def get_codebook_usage(self, encoding_indices: torch.Tensor) -> float:
        """
        Compute fraction of codebook used in current batch.
        
        Args:
            encoding_indices: Indices selected in current batch
            
        Returns:
            Usage fraction (0.0 to 1.0)
        """
        unique_codes = torch.unique(encoding_indices)
        usage = len(unique_codes) / self.num_embeddings
        return usage
    
    def compute_perplexity(self, encoding_indices: torch.Tensor) -> float:
        """
        Compute perplexity of codebook usage.
        
        Higher perplexity = more diverse usage.
        
        Args:
            encoding_indices: Indices selected in current batch
            
        Returns:
            Perplexity value
        """
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()


class Encoder(nn.Module):
    """
    Convolutional encoder with dropout regularization.
    
    Architecture:
    - Progressive downsampling with strided convolutions
    - Batch normalization for stability
    - Dropout for regularization
    - Final FC projection to embedding space
    """
    
    def __init__(self, input_dim: int, embedding_dim: int, n_conv_layers: int, dropout: float = 0.2):
        """
        Initialize encoder.
        
        Args:
            input_dim: Input dimension (B bins)
            embedding_dim: Output embedding dimension (D)
            n_conv_layers: Number of convolutional layers (2 or 3)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_conv_layers = n_conv_layers
        
        channels = MODEL_CONFIG['conv_channels']
        kernels = MODEL_CONFIG['kernel_sizes']
        strides = MODEL_CONFIG['strides']
        paddings = MODEL_CONFIG['paddings']
        
        layers = []
        
        # Conv layer 1: 1 -> 32 channels
        layers.extend([
            nn.Conv1d(1, channels[0], kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_channels = channels[0]
        current_length = (input_dim + 2*paddings[0] - kernels[0]) // strides[0] + 1
        
        # Conv layer 2: 32 -> 64 channels
        layers.extend([
            nn.Conv1d(channels[0], channels[1], kernel_size=kernels[1], stride=strides[1], padding=paddings[1]),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_channels = channels[1]
        current_length = (current_length + 2*paddings[1] - kernels[1]) // strides[1] + 1
        
        # Conv layer 3: 64 -> 128 channels (always present)
        layers.extend([
            nn.Conv1d(channels[1], channels[2], kernel_size=kernels[2], stride=strides[2], padding=paddings[2]),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_channels = channels[2]
        current_length = (current_length + 2*paddings[2] - kernels[2]) // strides[2] + 1
        
        # Optional conv layer 4: 128 -> 256 channels
        if n_conv_layers == 3:
            layers.extend([
                nn.Conv1d(channels[2], channels[3], kernel_size=kernels[3], stride=strides[3], padding=paddings[3]),
                nn.BatchNorm1d(channels[3]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_channels = channels[3]
            current_length = (current_length + 2*paddings[3] - kernels[3]) // strides[3] + 1
        
        self.conv_layers = nn.Sequential(*layers)
        self.flatten_size = current_channels * current_length
        self.fc = nn.Linear(self.flatten_size, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor (batch_size, B)
            
        Returns:
            z_e: Encoded representation (batch_size, D)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, B)
        x = self.conv_layers(x)  # (batch_size, C, L)
        x = x.view(x.size(0), -1)  # (batch_size, C*L)
        z_e = self.fc(x)  # (batch_size, D)
        return z_e


class Decoder(nn.Module):
    """
    Transposed convolutional decoder with dropout regularization.

    Architecture:
    - FC expansion from embedding space
    - Progressive upsampling with transposed convolutions
    - Batch normalization for stability
    - Dropout for regularization
    - Final projection to output dimension
    - Softmax activation to ensure output is a valid probability distribution
    """
    
    def __init__(
        self, 
        embedding_dim: int, 
        output_dim: int, 
        n_conv_layers: int, 
        encoder_flatten_size: int,
        dropout: float = 0.2
    ):
        """
        Initialize decoder.
        
        Args:
            embedding_dim: Input embedding dimension (D)
            output_dim: Output dimension (B bins)
            n_conv_layers: Number of convolutional layers (must match encoder)
            encoder_flatten_size: Flattened size from encoder (for reshape)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.n_conv_layers = n_conv_layers
        self.encoder_flatten_size = encoder_flatten_size
        
        channels = MODEL_CONFIG['conv_channels']
        kernels = MODEL_CONFIG['kernel_sizes']
        strides = MODEL_CONFIG['strides']
        paddings = MODEL_CONFIG['paddings']
        
        # Determine reshape dimensions
        if n_conv_layers == 3:
            self.reshape_channels = channels[3]
            self.reshape_length = encoder_flatten_size // channels[3]
        else:
            self.reshape_channels = channels[2]
            self.reshape_length = encoder_flatten_size // channels[2]
        
        # FC expansion
        self.fc = nn.Linear(embedding_dim, encoder_flatten_size)
        
        layers = []
        
        # Build decoder layers (reverse order of encoder)
        if n_conv_layers == 3:
            # ConvTranspose layer 4: 256 -> 128 channels
            layers.extend([
                nn.ConvTranspose1d(channels[3], channels[2], kernel_size=kernels[3], 
                                  stride=strides[3], padding=paddings[3], output_padding=1),
                nn.BatchNorm1d(channels[2]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # ConvTranspose layer 3: 128 -> 64 channels
        layers.extend([
            nn.ConvTranspose1d(channels[2], channels[1], kernel_size=kernels[2], 
                              stride=strides[2], padding=paddings[2], output_padding=1),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # ConvTranspose layer 2: 64 -> 32 channels
        layers.extend([
            nn.ConvTranspose1d(channels[1], channels[0], kernel_size=kernels[1], 
                              stride=strides[1], padding=paddings[1], output_padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # ConvTranspose layer 1: 32 -> 1 channel (final reconstruction)
        layers.append(
            nn.ConvTranspose1d(channels[0], 1, kernel_size=kernels[0], 
                              stride=strides[0], padding=paddings[0], output_padding=1)
        )
        
        self.conv_layers = nn.Sequential(*layers)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized vectors to reconstruct input probability distribution.

        Args:
            z_q: Quantized representation (batch_size, D)

        Returns:
            x_recon: Reconstructed probability distribution (batch_size, B)
                     Guaranteed to have non-negative values that sum to 1
        """
        x = self.fc(z_q)  # (batch_size, flatten_size)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)  # (batch_size, C, L)
        x = self.conv_layers(x)  # (batch_size, 1, B')
        x = x.squeeze(1)  # (batch_size, B')

        # Trim or pad to exact output dimension
        if x.size(1) > self.output_dim:
            x = x[:, :self.output_dim]
        elif x.size(1) < self.output_dim:
            padding = torch.zeros(x.size(0), self.output_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Apply softmax to ensure output is a valid probability distribution:
        # - All values are positive (exp function)
        # - All values sum to 1 (normalization)
        x = F.softmax(x, dim=1)

        return x


class VQVAEModel(nn.Module):
    """
    Complete VQ-VAE model for LOB representation learning.
    
    Architecture:
    - Encoder: LOB bins -> latent continuous space
    - Vector Quantizer: Continuous -> discrete codebook vectors
    - Decoder: Discrete codes -> reconstructed LOB bins
    """
    
    def __init__(self, config: Dict):
        """
        Initialize VQ-VAE model.

        Args:
            config: Hyperparameter configuration with keys:
                - B: Number of LOB bins
                - K: Codebook size
                - D: Embedding dimension
                - n_conv_layers: Number of conv layers (2 or 3)
                - dropout: Dropout probability
                - use_ema: Use EMA updates for codebook (optional, default: False)
                - ema_decay: EMA decay rate (optional, default: 0.99)
                - recon_loss_type: Reconstruction loss type ('mse' or 'wasserstein', default: 'wasserstein')
        """
        super().__init__()

        self.config = config

        B = config['B']
        K = config['K']
        D = config['D']
        n_conv_layers = config['n_conv_layers']
        dropout = config['dropout']
        use_ema = config.get('use_ema', False)
        ema_decay = config.get('ema_decay', 0.99)
        self.recon_loss_type = config.get('recon_loss_type', 'wasserstein')

        # Validate loss type
        if self.recon_loss_type not in ['mse', 'wasserstein']:
            raise ValueError(f"Invalid recon_loss_type: {self.recon_loss_type}. Use 'mse' or 'wasserstein'.")

        # Initialize components
        self.encoder = Encoder(B, D, n_conv_layers, dropout)
        self.vq = VectorQuantizer(K, D, use_ema=use_ema, ema_decay=ema_decay)
        self.decoder = Decoder(D, B, n_conv_layers, self.encoder.flatten_size, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input LOB vectors (batch_size, B)
            
        Returns:
            x_recon: Reconstructed LOB vectors (batch_size, B)
            loss_dict: Dictionary with individual loss components
        """
        # Encode
        z_e = self.encoder(x)

        # Quantize
        z_q, commitment_loss, codebook_loss, encoding_indices = self.vq(z_e)

        # Decode
        x_recon = self.decoder(z_q)

        # Compute reconstruction loss based on loss type
        if self.recon_loss_type == 'wasserstein':
            recon_loss = wasserstein_1d_loss(x_recon, x)
        else:  # mse
            recon_loss = F.mse_loss(x_recon, x)
        
        # Compute codebook usage
        codebook_usage = self.vq.get_codebook_usage(encoding_indices)
        perplexity = self.vq.compute_perplexity(encoding_indices)
        
        loss_dict = {
            'recon_loss': recon_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'encoding_indices': encoding_indices,
            'codebook_usage': codebook_usage,
            'perplexity': perplexity
        }
        
        return x_recon, loss_dict
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to discrete latent codes (for production inference).
        
        Args:
            x: Input LOB vectors (batch_size, B)
            
        Returns:
            encoding_indices: Discrete codebook indices (batch_size,)
        """
        with torch.no_grad():
            z_e = self.encoder(x)
            _, _, _, encoding_indices = self.vq(z_e)
        return encoding_indices