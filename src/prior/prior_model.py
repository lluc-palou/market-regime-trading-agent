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
    Causal CNN for joint modeling of latent codes and targets.

    Architecture:
    - Embedding layer for discrete codes (embedding_dim=64, matches VQ-VAE)
    - Target conditioning via concatenation (minimal params)
    - Stacked dilated causal convolutions with repeating dilation pattern
    - Residual connections with gated activations (WaveNet-style)
    - Dual output heads: codebook logits + target prediction

    Dilation strategy:
    - Repeating pattern [1, 2, 4, 8, 16, 32, 64] to control receptive field
    - Max RF = 127 timesteps (suitable for seq_len=120)
    - Depth achieved through repetition, not expansion
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int = 64,
        n_layers: int = 12,
        n_channels: int = 64,
        kernel_size: int = 2,
        dropout: float = 0.15,
        predict_target: bool = True
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.predict_target = predict_target

        # Embed discrete codes
        self.embedding = nn.Embedding(codebook_size, embedding_dim)

        # Initial projection to channel dimension (input: embedding + target scalar)
        input_dim = embedding_dim + 1 if predict_target else embedding_dim
        self.input_conv = nn.Conv1d(input_dim, n_channels, 1)

        # Dilated causal convolutions with repeating pattern for controlled RF
        # Pattern: [1, 2, 4, 8, 16, 32, 64] repeated to reach n_layers
        # This keeps RF ≤ 127 (suitable for seq_len=120) while adding depth
        base_dilations = [1, 2, 4, 8, 16, 32, 64]
        dilations = []
        for i in range(n_layers):
            dilations.append(base_dilations[i % len(base_dilations)])

        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                ResidualBlock(n_channels, kernel_size, dilation, dropout)
            )
        
        # Codebook output head
        self.codebook_output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_channels, codebook_size, 1)
        )

        # Target output head (minimal params)
        if predict_target:
            self.target_output = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(n_channels, n_channels // 2, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(n_channels // 2, 1, 1)
            )
    
    def forward(self, z_seq, target_seq=None):
        """
        Forward pass with optional target conditioning.

        Args:
            z_seq: (batch, seq_len) discrete codes
            target_seq: (batch, seq_len) target values (required if predict_target=True)

        Returns:
            If predict_target=True:
                codebook_logits: (batch, seq_len, codebook_size)
                target_preds: (batch, seq_len)
            Else:
                codebook_logits: (batch, seq_len, codebook_size)
        """
        # Embed: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        x = self.embedding(z_seq)

        # Concatenate target if predicting
        if self.predict_target:
            if target_seq is None:
                raise ValueError("target_seq required when predict_target=True")
            # Add target as extra feature: (batch, seq_len, 1)
            target_feat = target_seq.unsqueeze(-1)
            # Concatenate: (batch, seq_len, embedding_dim + 1)
            x = torch.cat([x, target_feat], dim=-1)

        # Transpose for conv: (batch, embedding_dim+1, seq_len) or (batch, embedding_dim, seq_len)
        x = x.transpose(1, 2)

        # Initial projection
        x = self.input_conv(x)

        # Dilated causal convolutions
        for layer in self.layers:
            x = layer(x)

        # Codebook output: (batch, codebook_size, seq_len)
        codebook_logits = self.codebook_output(x)
        codebook_logits = codebook_logits.transpose(1, 2)  # (batch, seq_len, codebook_size)

        if self.predict_target:
            # Target output: (batch, 1, seq_len)
            target_preds = self.target_output(x)
            target_preds = target_preds.squeeze(1)  # (batch, seq_len)
            return codebook_logits, target_preds
        else:
            return codebook_logits
    
    @torch.no_grad()
    def sample(self, n_samples: int, seq_len: int, temperature: float = 1.0, device=None):
        """
        Autoregressively generate latent sequences and targets.

        Args:
            n_samples: Number of sequences to generate
            seq_len: Length of each sequence
            temperature: Sampling temperature (1.0 = standard)
            device: torch device

        Returns:
            If predict_target=True:
                z_seq: (n_samples, seq_len) of discrete codes
                target_seq: (n_samples, seq_len) of predicted targets
            Else:
                z_seq: (n_samples, seq_len) of discrete codes
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Initialize with random start codes
        z_seq = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
        z_seq[:, 0] = torch.randint(0, self.codebook_size, (n_samples,), device=device)

        if self.predict_target:
            # Initialize target sequence with small random values
            target_seq = torch.zeros(n_samples, seq_len, device=device)
            target_seq[:, 0] = torch.randn(n_samples, device=device) * 0.0001  # Small init

            # Autoregressively generate both
            for t in range(1, seq_len):
                # Forward pass on history
                codebook_logits, target_preds = self.forward(
                    z_seq[:, :t], target_seq[:, :t]
                )

                # Sample codebook index with temperature
                probs = F.softmax(codebook_logits[:, -1, :] / temperature, dim=-1)
                z_seq[:, t] = torch.multinomial(probs, 1).squeeze(1)

                # Use predicted target
                target_seq[:, t] = target_preds[:, -1]

            return z_seq, target_seq
        else:
            # Original behavior: only generate codes
            for t in range(1, seq_len):
                logits = self.forward(z_seq[:, :t])[:, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)
                z_seq[:, t] = torch.multinomial(probs, 1).squeeze(1)

            return z_seq
    
    def compute_receptive_field(self):
        """
        Compute theoretical receptive field of the network.

        Uses repeating dilation pattern: [1, 2, 4, 8, 16, 32, 64]
        Maximum RF = 127 timesteps (covers seq_len=120)
        """
        base_dilations = [1, 2, 4, 8, 16, 32, 64]
        dilations = []
        for i in range(self.n_layers):
            dilations.append(base_dilations[i % len(base_dilations)])

        receptive_field = 1
        for dilation in dilations:
            receptive_field += (2 - 1) * dilation  # kernel_size=2

        return receptive_field