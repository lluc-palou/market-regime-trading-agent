"""
Prior Model Architecture

Causal CNN for modeling latent code distributions.
WaveNet-style architecture with dilated convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal convolution with left padding to prevent future leakage."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )
    
    def forward(self, x):
        # Pad on the left (past) only
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with gated activation."""
    
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv_tanh = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv_sigmoid = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        # Gated activation (WaveNet style)
        tanh_out = torch.tanh(self.conv_tanh(x))
        sigmoid_out = torch.sigmoid(self.conv_sigmoid(x))
        gated = tanh_out * sigmoid_out
        
        # Projection and residual
        out = self.projection(self.dropout(gated))
        return x + out


class LatentPriorCNN(nn.Module):
    """
    Causal CNN for modeling prior distribution of latent codes.
    
    Architecture:
    - Embedding layer for discrete codes
    - Stacked dilated causal convolutions with residual connections
    - Output projection to codebook size
    """
    
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int = 128,
        n_layers: int = 10,
        n_channels: int = 128,
        kernel_size: int = 2,
        dropout: float = 0.15
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Embed discrete codes
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        
        # Initial projection to channel dimension
        self.input_conv = nn.Conv1d(embedding_dim, n_channels, 1)
        
        # Dilated causal convolutions (exponentially increasing dilation)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(
                ResidualBlock(n_channels, kernel_size, dilation, dropout)
            )
        
        # Output projection to codebook logits
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_channels, codebook_size, 1)
        )
    
    def forward(self, z_seq):
        """
        Forward pass.
        
        Args:
            z_seq: (batch, seq_len) discrete codes
            
        Returns:
            logits: (batch, seq_len, codebook_size)
        """
        # Embed: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        x = self.embedding(z_seq)
        
        # Transpose for conv: (batch, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Initial projection
        x = self.input_conv(x)
        
        # Dilated causal convolutions
        for layer in self.layers:
            x = layer(x)
        
        # Output projection: (batch, codebook_size, seq_len)
        x = self.output_conv(x)
        
        # Transpose back: (batch, seq_len, codebook_size)
        return x.transpose(1, 2)
    
    @torch.no_grad()
    def sample(self, n_samples: int, seq_len: int, temperature: float = 1.0, device=None):
        """
        Autoregressively generate latent sequences.
        
        Args:
            n_samples: Number of sequences to generate
            seq_len: Length of each sequence
            temperature: Sampling temperature (1.0 = standard)
            device: torch device
            
        Returns:
            z_seq: (n_samples, seq_len) of discrete codes
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Initialize with random start codes
        z_seq = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
        z_seq[:, 0] = torch.randint(0, self.codebook_size, (n_samples,), device=device)
        
        # Autoregressively generate
        for t in range(1, seq_len):
            # Forward pass on history
            logits = self.forward(z_seq[:, :t])[:, -1, :]  # (n_samples, codebook_size)
            
            # Sample with temperature
            probs = F.softmax(logits / temperature, dim=-1)
            z_seq[:, t] = torch.multinomial(probs, 1).squeeze(1)
        
        return z_seq
    
    def compute_receptive_field(self):
        """Compute theoretical receptive field of the network."""
        receptive_field = 1
        for i in range(self.n_layers):
            dilation = 2 ** i
            receptive_field += (2 - 1) * dilation  # kernel_size=2
        return receptive_field