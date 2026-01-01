"""Configuration classes for RL agent training."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class ExperimentType(Enum):
    """Experiment types for PPO training."""
    EXP1_BOTH_ORIGINAL = 1      # Both codebook + features, train/val on original
    EXP2_FEATURES_ORIGINAL = 2  # Features only, train/val on original
    EXP3_CODEBOOK_ORIGINAL = 3  # Codebook only, train/val on original


@dataclass
class VQVAEConfig:
    """VQ-VAE production architecture specification."""
    B: int = 1001                        # LOB dimension
    D: int = 64                          # Latent dimension
    K: int = 128                         # Codebook size
    beta: float = 0.5                    # Commitment cost
    dropout: float = 0.2
    lr: float = 0.001
    n_conv_layers: int = 3


@dataclass
class ModelConfig:
    """Transformer architecture configuration."""
    vocab_size: int = 128                # VQ-VAE codebook size (K)
    n_features: int = 18                 # Number of hand-crafted features
    d_codebook: int = 80                 # Codebook embedding dimension (increased from 64)
    d_features: int = 80                 # Feature projection dimension (increased from 64)
    d_model: int = 160                   # Transformer model dimension (increased from 128)
    n_heads: int = 8                     # Number of attention heads (increased from 4)
    n_layers: int = 4                    # Number of transformer layers (increased from 2)
    dropout: float = 0.2                 # Dropout rate (increased for regularization)
    window_size: int = 10                # Observation window (W samples) - matches horizon for temporal context
    horizon: int = 10                    # Reward horizon - cumulative forward returns over H steps
    min_log_std: float = -20.0           # Minimum log std for policy
    max_log_std: float = 2.0             # Maximum log std for policy
    ffn_expansion: int = 4               # FFN dimension = d_model × ffn_expansion
    
    def count_parameters(self) -> int:
        """Estimate total model parameters."""
        # Codebook embedding
        codebook_params = self.vocab_size * self.d_codebook
        
        # Feature projection
        feature_params = self.n_features * self.d_features + self.d_features
        
        # Fusion layer
        fusion_input = self.d_codebook + self.d_features
        fusion_params = fusion_input * self.d_model + self.d_model
        
        # Transformer layers
        dim_ff = self.d_model * self.ffn_expansion
        params_per_layer = (
            # Attention (Q, K, V projections + output)
            4 * (self.d_model * self.d_model + self.d_model) +
            # FFN (two layers)
            (self.d_model * dim_ff + dim_ff) +
            (dim_ff * self.d_model + self.d_model) +
            # LayerNorm (2x)
            2 * (2 * self.d_model)
        )
        transformer_params = self.n_layers * params_per_layer
        
        # Heads
        head_params = 3 * (self.d_model + 1)  # mean, logstd, critic
        
        total = (codebook_params + feature_params + fusion_params + 
                transformer_params + head_params)
        
        return int(total)


@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    learning_rate: float = 1e-4          # Adam learning rate
    weight_decay: float = 1e-3           # L2 regularization
    gamma: float = 0.95                  # Discount factor
    gae_lambda: float = 0.95             # GAE lambda parameter
    clip_ratio: float = 0.1              # PPO clipping parameter (reduced for stability with high variance)
    value_coef: float = 1.0              # Value loss coefficient (increased for better value estimates)
    entropy_coef: float = 0.03           # Fixed entropy coefficient (encourages exploration, increased for moderate exploration)
    uncertainty_coef: float = 0.1        # Uncertainty penalty coefficient (prevents std exploitation)
    activity_coef: float = 0.0005        # Inactivity penalty coefficient (reduced to learn quality over quantity)
    max_grad_norm: float = 0.5           # Gradient clipping norm
    n_epochs: int = 2                    # PPO epochs per update (reduced for speed)
    batch_size: int = 256                # Minibatch size (increased 8x to utilize GPU)
    buffer_capacity: int = 2048          # Trajectory buffer size (increased 4x)
    

@dataclass
class RewardConfig:
    """Reward function parameters."""
    spread_bps: float = 5.0              # Spread cost in basis points
    tc_bps: float = 2.5                  # Transaction cost in basis points
    lambda_risk: float = 1.0             # Risk adjustment weight
    alpha_penalty: float = 0.01          # Position size penalty
    epsilon: float = 1e-8                # Numerical stability


@dataclass
class TrainingConfig:
    """Training procedure configuration."""
    max_epochs: int = 50                 # Maximum training epochs (increased from 10)
    patience: int = 10                   # Early stopping patience (increased to utilize 50 epochs better)
    min_delta: float = 0.01              # Minimum improvement for early stopping
    validate_every: int = 1              # Validate every N epochs
    log_every: int = 10                  # Log every N episodes
    checkpoint_dir: str = "checkpoints"  # Directory for model checkpoints
    log_dir: str = "logs"                # Directory for logs
    seed: int = 42                       # Random seed
    device: str = "cuda"                 # Device (cuda or cpu)


@dataclass
class DataConfig:
    """Data loading configuration."""
    mongodb_uri: str = "mongodb://localhost:27017/"
    database_name: str = "raw"  # Match pipeline convention
    split_ids: List[int] = field(default_factory=lambda: list(range(4)))  # Splits to train
    role_train: str = "train"            # Training role filter
    role_val: str = "val"                # Validation role filter
    stream_episodes: bool = True         # Stream episodes on-demand
    experiment_type: ExperimentType = ExperimentType.EXP1_BOTH_ORIGINAL  # Experiment type


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "rl_agent_experiment"
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data_dict = self.data.__dict__.copy()
        data_dict['experiment_type'] = self.data.experiment_type.value  # Convert enum to int

        return {
            "name": self.name,
            "vqvae": self.vqvae.__dict__,
            "model": self.model.__dict__,
            "ppo": self.ppo.__dict__,
            "reward": self.reward.__dict__,
            "training": self.training.__dict__,
            "data": data_dict,
            "total_parameters": self.model.count_parameters()
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data.get("name", "experiment"),
            vqvae=VQVAEConfig(**data.get("vqvae", {})),
            model=ModelConfig(**data.get("model", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            reward=RewardConfig(**data.get("reward", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {}))
        )


def get_hyperparameter_grid() -> List[ModelConfig]:
    """
    Get pre-defined hyperparameter search grid.
    
    Target: 12 configurations within [88.5k, 354k] parameters (0.5x to 2x of 177k)
    Based on VQ-VAE production: K=128, D=64
    
    Returns:
        List of 12 ModelConfig objects
    """
    configs = []
    
    # Config 1: Smallest (0.51x baseline) - 91k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=32, d_features=32,
        d_model=64, n_heads=2, n_layers=1,
        dropout=0.1, ffn_expansion=4, window_size=50, horizon=10
    ))
    
    # Config 2: Very small (0.61x baseline) - 109k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=40, d_features=40,
        d_model=80, n_heads=2, n_layers=1,
        dropout=0.1, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 3: Small (0.81x baseline) - 144k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=48, d_features=48,
        d_model=96, n_heads=4, n_layers=1,
        dropout=0.15, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 4: Small-medium (0.95x baseline) - 169k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=56, d_features=56,
        d_model=112, n_heads=4, n_layers=1,
        dropout=0.15, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 5: Medium (1.11x baseline) - 196k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=48, d_features=48,
        d_model=96, n_heads=4, n_layers=2,
        dropout=0.15, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 6: Medium (1.40x baseline) - 248k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=56, d_features=56,
        d_model=112, n_heads=4, n_layers=2,
        dropout=0.2, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 7: Medium (1.44x baseline) - 254k params - compact FFN
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=64, d_features=64,
        d_model=128, n_heads=8, n_layers=2,
        dropout=0.25, ffn_expansion=3, window_size=50, horizon=10
    ))

    # Config 8: Medium-large (1.57x baseline) - 277k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=60, d_features=60,
        d_model=120, n_heads=4, n_layers=2,
        dropout=0.2, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 9: Medium-large (1.61x baseline) - 285k params - compact FFN
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=72, d_features=72,
        d_model=144, n_heads=4, n_layers=2,
        dropout=0.2, ffn_expansion=3, window_size=50, horizon=10
    ))

    # Config 10: Large single-layer (1.65x baseline) - 292k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=80, d_features=80,
        d_model=160, n_heads=4, n_layers=1,
        dropout=0.2, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 11: Large (1.74x baseline) - 307k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=64, d_features=64,
        d_model=128, n_heads=4, n_layers=2,
        dropout=0.2, ffn_expansion=4, window_size=50, horizon=10
    ))

    # Config 12: Largest (1.93x baseline) - 341k params
    configs.append(ModelConfig(
        vocab_size=128, n_features=18,
        d_codebook=68, d_features=68,
        d_model=136, n_heads=4, n_layers=2,
        dropout=0.2, ffn_expansion=4, window_size=50, horizon=10
    ))
    
    return configs


def print_config_summary(configs: List[ModelConfig]):
    """Print summary table of configurations."""
    print("\n" + "=" * 110)
    print("HYPERPARAMETER SEARCH GRID")
    print("=" * 110)
    print(f"{'#':<4} {'d_emb':<8} {'d_model':<10} {'n_heads':<10} {'n_layers':<10} "
          f"{'FFN':<8} {'dropout':<10} {'Parameters':<15} {'Ratio':<10}")
    print("-" * 110)
    
    for i, config in enumerate(configs, 1):
        params = config.count_parameters()
        ratio = params / 177000
        d_emb = config.d_codebook  # Same as d_features
        print(f"{i:<4} {d_emb:<8} {config.d_model:<10} {config.n_heads:<10} {config.n_layers:<10} "
              f"{config.ffn_expansion}x{'':<6} {config.dropout:<10.2f} {params:<15,} {ratio:<10.2f}")
    
    print("=" * 110)
    print(f"Total configurations: {len(configs)}")
    min_p = min(c.count_parameters() for c in configs)
    max_p = max(c.count_parameters() for c in configs)
    print(f"Parameter range: [{min_p:,}, {max_p:,}]")
    print(f"Target range: [88,500, 354,000]")
    print(f"Within bounds: {'✓ YES' if min_p >= 88500 and max_p <= 354000 else '✗ NO'}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    # Generate and display hyperparameter grid
    configs = get_hyperparameter_grid()
    print_config_summary(configs)
    
    # VQ-VAE info
    vqvae = VQVAEConfig()
    print("VQ-VAE Production Architecture:")
    print(f"  Codebook size (K): {vqvae.K}")
    print(f"  Latent dimension (D): {vqvae.D}")
    print(f"  Input dimension (B): {vqvae.B}")
    print(f"  Conv layers: {vqvae.n_conv_layers}")
    print(f"  Dropout: {vqvae.dropout}")
    print(f"  Beta: {vqvae.beta}")
    print()
    
    # Save configurations
    import os
    os.makedirs("configs", exist_ok=True)
    
    for i, model_config in enumerate(configs, 1):
        exp_config = ExperimentConfig(
            name=f"config_{i:02d}",
            model=model_config
        )
        exp_config.save(f"configs/config_{i:02d}.json")
    
    print(f"✓ Saved {len(configs)} configurations to 'configs/' directory")