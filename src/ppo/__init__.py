"""RL Agent module for limit order book trading."""

from .model import ActorCriticTransformer, ActorCriticFeatures, ActorCriticCodebook
from .buffer import StateBuffer, TrajectoryBuffer, AgentState, Transition
from .ppo import ppo_update, compute_gae
from .environment import Episode, EpisodeLoader, get_valid_timesteps
from .reward import (
    compute_policy_based_position,
    compute_volatility_scaled_position,
    compute_simple_reward,
    compute_transaction_cost,
    compute_unrealized_pnl
)
from .utils import (
    MetricsLogger,
    compute_sharpe_ratio,
    save_checkpoint,
    load_checkpoint,
    print_metrics
)
from .config import (
    ModelConfig,
    PPOConfig,
    RewardConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig
)

__all__ = [
    'ActorCriticTransformer',
    'ActorCriticFeatures',
    'ActorCriticCodebook',
    'StateBuffer',
    'TrajectoryBuffer',
    'AgentState',
    'Transition',
    'ppo_update',
    'compute_gae',
    'Episode',
    'EpisodeLoader',
    'get_valid_timesteps',
    'compute_policy_based_position',
    'compute_volatility_scaled_position',
    'compute_simple_reward',
    'compute_transaction_cost',
    'compute_unrealized_pnl',
    'MetricsLogger',
    'compute_sharpe_ratio',
    'save_checkpoint',
    'load_checkpoint',
    'print_metrics',
    'ModelConfig',
    'PPOConfig',
    'RewardConfig',
    'TrainingConfig',
    'DataConfig',
    'ExperimentConfig'
]