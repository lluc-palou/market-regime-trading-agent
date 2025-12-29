"""Actor-Critic Transformer architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Timestamp-aware positional encoding."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input embeddings
            timestamps: (batch, seq_len) Unix timestamps in seconds
        
        Returns:
            x + positional encoding
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize timestamps to [0, 1] within each sequence
        ts_min = timestamps.min(dim=1, keepdim=True)[0]
        ts_max = timestamps.max(dim=1, keepdim=True)[0]
        timestamps_normalized = (timestamps - ts_min) / (ts_max - ts_min + 1e-8)
        
        # Create sinusoidal encoding
        position = timestamps_normalized.unsqueeze(-1)  # (batch, seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32) *
            -(np.log(10000.0) / self.d_model)
        )
        
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        
        return x + pe


class ActorCriticTransformer(nn.Module):
    """Transformer-based actor-critic for trading - Experiment 1.

    Architecture aligned with VQ-VAE production config:
        - vocab_size=128 (VQ-VAE codebook size K)
        - Processes codebook indices + hand-crafted features (BOTH sources)
        - Causal transformer for temporal modeling
        - Gaussian policy for continuous actions
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.window_size = config.window_size
        
        # Input embeddings
        self.codebook_embedding = nn.Embedding(config.vocab_size, config.d_codebook)
        self.feature_projection = nn.Linear(config.n_features, config.d_features)
        
        # Fusion layer (combines codebook + features)
        fusion_input_dim = config.d_codebook + config.d_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.d_model)
        
        # Transformer encoder with causal masking
        dim_feedforward = config.d_model * config.ffn_expansion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False  # Disabled because norm_first=True
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(config.d_model, 1)
        self.actor_logstd = nn.Linear(config.d_model, 1)
        
        # Critic head (value function)
        self.critic = nn.Linear(config.d_model, 1)
        
        # Register buffer for causal mask
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(config.window_size)
        )
    
    def forward(
        self,
        codebooks: torch.Tensor,
        features: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            codebooks: (batch, seq_len) integer tensor [0, 127] from VQ-VAE
            features: (batch, seq_len, n_features) float tensor
            timestamps: (batch, seq_len) float tensor (Unix timestamps)
        
        Returns:
            mean: (batch, 1) action mean
            log_std: (batch, 1) action log std
            value: (batch, 1) state value
        """
        batch_size, seq_len = codebooks.shape
        
        # Embed codebooks (lookup table)
        codebook_emb = self.codebook_embedding(codebooks)  # (batch, seq_len, d_codebook)
        
        # Project features (linear layer)
        features_proj = self.feature_projection(features)  # (batch, seq_len, d_features)
        
        # Fuse embeddings
        fused = torch.cat([codebook_emb, features_proj], dim=-1)
        fused = self.fusion(fused)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        fused = self.positional_encoding(fused, timestamps)
        
        # Create causal mask for current sequence length
        if seq_len != self.window_size:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=codebooks.device
            )
        else:
            causal_mask = self.causal_mask
        
        # Transformer encoding (causal attention)
        encoded = self.transformer(fused, mask=causal_mask)  # (batch, seq_len, d_model)
        
        # Use last token for decision
        last_hidden = encoded[:, -1, :]  # (batch, d_model)
        
        # Actor: policy distribution parameters
        mean = self.actor_mean(last_hidden)
        log_std = self.actor_logstd(last_hidden)
        log_std = torch.clamp(log_std, self.config.min_log_std, self.config.max_log_std)
        
        # Critic: state value
        value = self.critic(last_hidden)
        
        return mean, log_std, value
    
    def act(
        self,
        codebooks: torch.Tensor,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given current state.

        Args:
            codebooks: (seq_len,) or (batch, seq_len)
            features: (seq_len, n_features) or (batch, seq_len, n_features)
            timestamps: (seq_len,) or (batch, seq_len)
            deterministic: If True, return mean action (for evaluation)

        Returns:
            action: sampled action in [-1, 1]
            log_prob: log probability of action
            value: estimated state value
            std: policy standard deviation (agent's uncertainty)
        """
        # Add batch dimension if needed
        if codebooks.dim() == 1:
            codebooks = codebooks.unsqueeze(0)
            features = features.unsqueeze(0)
            timestamps = timestamps.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Forward pass
        mean, log_std, value = self.forward(codebooks, features, timestamps)

        # Compute std (always needed for position sizing)
        std = torch.exp(log_std)

        if deterministic:
            # Use mean action (for evaluation)
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action)  # Not used in evaluation
        else:
            # Sample from Gaussian policy
            normal = torch.distributions.Normal(mean, std)
            action_unbounded = normal.rsample()  # Reparameterization trick

            # Apply tanh squashing
            action = torch.tanh(action_unbounded)

            # Compute log prob with Jacobian correction
            log_prob = normal.log_prob(action_unbounded)
            log_prob -= torch.log(1 - action ** 2 + 1e-6)

        if squeeze_output:
            return action.squeeze(), log_prob.squeeze(), value.squeeze(), std.squeeze()
        else:
            return action, log_prob, value, std
    
    def evaluate_actions(
        self,
        codebooks: torch.Tensor,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs and values for batch of states/actions (for PPO update).

        Args:
            codebooks: (batch, seq_len)
            features: (batch, seq_len, n_features)
            timestamps: (batch, seq_len)
            actions: (batch,) actions in [-1, 1]

        Returns:
            log_probs: (batch,) log probabilities
            values: (batch,) state values
            entropy: (batch,) policy entropy
            std: (batch,) action standard deviation
        """
        # Forward pass
        mean, log_std, value = self.forward(codebooks, features, timestamps)

        # Compute log probs
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean.squeeze(-1), std.squeeze(-1))

        # Inverse tanh to get unbounded actions
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        actions_unbounded = torch.atanh(actions_clamped)

        log_prob = normal.log_prob(actions_unbounded)
        log_prob -= torch.log(1 - actions ** 2 + 1e-6)

        # Compute entropy
        entropy = normal.entropy()

        return log_prob, value.squeeze(-1), entropy, std.squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ActorCriticFeatures(nn.Module):
    """Transformer-based actor-critic using ONLY features - Experiment 2.

    Architecture using only hand-crafted features (no codebook):
        - Processes only 18 hand-crafted features
        - Causal transformer for temporal modeling
        - Gaussian policy for continuous actions
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.window_size = config.window_size

        # Input projection (features only)
        self.feature_projection = nn.Linear(config.n_features, config.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.d_model)

        # Transformer encoder with causal masking
        dim_feedforward = config.d_model * config.ffn_expansion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False  # Disabled because norm_first=True
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(config.d_model, 1)
        self.actor_logstd = nn.Linear(config.d_model, 1)

        # Critic head (value function)
        self.critic = nn.Linear(config.d_model, 1)

        # Register buffer for causal mask
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(config.window_size)
        )

    def forward(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using only features.

        Args:
            features: (batch, seq_len, n_features) float tensor
            timestamps: (batch, seq_len) float tensor (Unix timestamps)

        Returns:
            mean: (batch, 1) action mean
            log_std: (batch, 1) action log std
            value: (batch, 1) state value
        """
        batch_size, seq_len, _ = features.shape

        # Project features
        features_proj = self.feature_projection(features)  # (batch, seq_len, d_model)

        # Add positional encoding
        features_proj = self.positional_encoding(features_proj, timestamps)

        # Create causal mask for current sequence length
        if seq_len != self.window_size:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=features.device
            )
        else:
            causal_mask = self.causal_mask

        # Transformer encoding (causal attention)
        encoded = self.transformer(features_proj, mask=causal_mask)

        # Use last token for decision
        last_hidden = encoded[:, -1, :]

        # Actor: policy distribution parameters
        mean = self.actor_mean(last_hidden)
        log_std = self.actor_logstd(last_hidden)
        log_std = torch.clamp(log_std, self.config.min_log_std, self.config.max_log_std)

        # Critic: state value
        value = self.critic(last_hidden)

        return mean, log_std, value

    def act(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given current state (features only).

        Args:
            features: (seq_len, n_features) or (batch, seq_len, n_features)
            timestamps: (seq_len,) or (batch, seq_len)
            deterministic: If True, return mean action

        Returns:
            action: sampled action in [-1, 1]
            log_prob: log probability of action
            value: estimated state value
            std: policy standard deviation (agent's uncertainty)
        """
        # Add batch dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(0)
            timestamps = timestamps.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Forward pass
        mean, log_std, value = self.forward(features, timestamps)

        # Compute std (always needed for position sizing)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action)
        else:
            normal = torch.distributions.Normal(mean, std)
            action_unbounded = normal.rsample()
            action = torch.tanh(action_unbounded)
            log_prob = normal.log_prob(action_unbounded)
            log_prob -= torch.log(1 - action ** 2 + 1e-6)

        if squeeze_output:
            return action.squeeze(), log_prob.squeeze(), value.squeeze(), std.squeeze()
        else:
            return action, log_prob, value, std

    def evaluate_actions(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs and values for batch of states/actions (for PPO update).

        Args:
            features: (batch, seq_len, n_features)
            timestamps: (batch, seq_len)
            actions: (batch,) actions in [-1, 1]

        Returns:
            log_probs: (batch,) log probabilities
            values: (batch,) state values
            entropy: (batch,) policy entropy
            std: (batch,) action standard deviation
        """
        # Forward pass
        mean, log_std, value = self.forward(features, timestamps)

        # Compute log probs
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean.squeeze(-1), std.squeeze(-1))

        # Inverse tanh to get unbounded actions
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        actions_unbounded = torch.atanh(actions_clamped)

        log_prob = normal.log_prob(actions_unbounded)
        log_prob -= torch.log(1 - actions ** 2 + 1e-6)

        # Compute entropy
        entropy = normal.entropy()

        return log_prob, value.squeeze(-1), entropy, std.squeeze(-1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ActorCriticCodebook(nn.Module):
    """Transformer-based actor-critic using ONLY codebook - Experiment 3.

    Architecture using only VQ-VAE codebook indices:
        - vocab_size=128 (VQ-VAE codebook size K)
        - Processes only codebook indices (no hand-crafted features)
        - Causal transformer for temporal modeling
        - Gaussian policy for continuous actions
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.window_size = config.window_size

        # Input embedding (codebook only)
        self.codebook_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.d_model)

        # Transformer encoder with causal masking
        dim_feedforward = config.d_model * config.ffn_expansion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False  # Disabled because norm_first=True
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(config.d_model, 1)
        self.actor_logstd = nn.Linear(config.d_model, 1)

        # Critic head (value function)
        self.critic = nn.Linear(config.d_model, 1)

        # Register buffer for causal mask
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(config.window_size)
        )

    def forward(
        self,
        codebooks: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using only codebook indices.

        Args:
            codebooks: (batch, seq_len) integer tensor [0, 127] from VQ-VAE
            timestamps: (batch, seq_len) float tensor (Unix timestamps)

        Returns:
            mean: (batch, 1) action mean
            log_std: (batch, 1) action log std
            value: (batch, 1) state value
        """
        batch_size, seq_len = codebooks.shape

        # Embed codebooks
        codebook_emb = self.codebook_embedding(codebooks)  # (batch, seq_len, d_model)

        # Add positional encoding
        codebook_emb = self.positional_encoding(codebook_emb, timestamps)

        # Create causal mask for current sequence length
        if seq_len != self.window_size:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=codebooks.device
            )
        else:
            causal_mask = self.causal_mask

        # Transformer encoding (causal attention)
        encoded = self.transformer(codebook_emb, mask=causal_mask)

        # Use last token for decision
        last_hidden = encoded[:, -1, :]

        # Actor: policy distribution parameters
        mean = self.actor_mean(last_hidden)
        log_std = self.actor_logstd(last_hidden)
        log_std = torch.clamp(log_std, self.config.min_log_std, self.config.max_log_std)

        # Critic: state value
        value = self.critic(last_hidden)

        return mean, log_std, value

    def act(
        self,
        codebooks: torch.Tensor,
        timestamps: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given current state (codebook only).

        Args:
            codebooks: (seq_len,) or (batch, seq_len)
            timestamps: (seq_len,) or (batch, seq_len)
            deterministic: If True, return mean action

        Returns:
            action: sampled action in [-1, 1]
            log_prob: log probability of action
            value: estimated state value
            std: policy standard deviation (agent's uncertainty)
        """
        # Add batch dimension if needed
        if codebooks.dim() == 1:
            codebooks = codebooks.unsqueeze(0)
            timestamps = timestamps.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Forward pass
        mean, log_std, value = self.forward(codebooks, timestamps)

        # Compute std (always needed for position sizing)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action)
        else:
            normal = torch.distributions.Normal(mean, std)
            action_unbounded = normal.rsample()
            action = torch.tanh(action_unbounded)
            log_prob = normal.log_prob(action_unbounded)
            log_prob -= torch.log(1 - action ** 2 + 1e-6)

        if squeeze_output:
            return action.squeeze(), log_prob.squeeze(), value.squeeze(), std.squeeze()
        else:
            return action, log_prob, value, std

    def evaluate_actions(
        self,
        codebooks: torch.Tensor,
        timestamps: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs and values for batch of states/actions (for PPO update).

        Args:
            codebooks: (batch, seq_len)
            timestamps: (batch, seq_len)
            actions: (batch,) actions in [-1, 1]

        Returns:
            log_probs: (batch,) log probabilities
            values: (batch,) state values
            entropy: (batch,) policy entropy
            std: (batch,) action standard deviation
        """
        # Forward pass
        mean, log_std, value = self.forward(codebooks, timestamps)

        # Compute log probs
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean.squeeze(-1), std.squeeze(-1))

        # Inverse tanh to get unbounded actions
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        actions_unbounded = torch.atanh(actions_clamped)

        log_prob = normal.log_prob(actions_unbounded)
        log_prob -= torch.log(1 - actions ** 2 + 1e-6)

        # Compute entropy
        entropy = normal.entropy()

        return log_prob, value.squeeze(-1), entropy, std.squeeze(-1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)