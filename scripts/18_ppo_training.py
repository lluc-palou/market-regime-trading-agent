"""
PPO Agent Training Script (Stage 18)

Trains PPO agents for limit order book trading using VQ-VAE latent representations
and hand-crafted features.

This is Stage 18 in the pipeline - follows synthetic generation (Stage 17) or can use
VQ-VAE production training output (Stage 14) directly.

Input: split_X_input collections in database 'raw' with:
       - codebook: VQ-VAE latent codes [0-127]
       - features: hand-crafted features (18 dimensions)
       - target: forward returns
       - timestamp: Unix timestamps
       - role: 'train' or 'validation'
       - fold_id: fold identifier

Output: Trained PPO agents saved to artifacts/ppo_agents/
        MLflow tracking with training metrics
        Checkpoints for best models per split

Usage:
    python scripts/18_ppo_training.py
"""

import os
import sys
from pathlib import Path
import time

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# =================================================================================================
# Unicode/MLflow Fix for Windows - MUST BE FIRST!
# =================================================================================================
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
    os.environ['PYTHONUTF8'] = '1'

    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

# Patch MLflow emoji issue
try:
    from mlflow.tracking._tracking_service import client as mlflow_client

    _original_log_url = mlflow_client.TrackingServiceClient._log_url

    def _patched_log_url(self, run_id):
        try:
            run = self.get_run(run_id)
            run_name = run.info.run_name or run_id
            run_url = self._get_run_url(run.info.experiment_id, run_id)
            sys.stdout.write(f"[RUN] View run {run_name} at: {run_url}\n")
            sys.stdout.flush()
        except:
            pass

    mlflow_client.TrackingServiceClient._log_url = _patched_log_url
except:
    pass
# =================================================================================================

import torch
import torch.optim as optim
import mlflow
import numpy as np
from datetime import datetime
from pymongo import MongoClient, ASCENDING

from src.utils.logging import logger
from src.ppo import (
    ActorCriticTransformer,
    ActorCriticFeatures,
    ActorCriticCodebook,
    EpisodeLoader,
    StateBuffer,
    TrajectoryBuffer,
    AgentState,
    Transition,
    ppo_update,
    get_valid_timesteps,
    compute_volatility_scaled_position,
    compute_simple_reward,
    compute_unrealized_pnl,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    MetricsLogger,
    compute_sharpe_ratio,
    save_checkpoint,
    print_metrics
)
from src.ppo.config import ExperimentType

# =================================================================================================
# Configuration
# =================================================================================================

DB_NAME = "raw"
COLLECTION_PREFIX = "split_"
COLLECTION_SUFFIX = "_input"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "PPO_Agent_Training"

MONGO_URI = "mongodb://127.0.0.1:27017/"

# Artifact directories
ARTIFACT_BASE_DIR = Path(REPO_ROOT) / "artifacts" / "ppo_agents"
CHECKPOINT_DIR = ARTIFACT_BASE_DIR / "checkpoints"
LOG_DIR = ARTIFACT_BASE_DIR / "logs"

# Training configuration
WINDOW_SIZE = 50  # Observation window (W samples)
HORIZON = 10      # Reward horizon (H samples)
EPISODE_CHUNK_SIZE = 120  # Split day episodes into hourly chunks (120 samples ≈ 1 hour)

# =================================================================================================
# Helper Functions
# =================================================================================================

def discover_splits(mongo_uri: str, db_name: str, collection_prefix: str, collection_suffix: str):
    """
    Discover available split IDs from MongoDB.

    Returns:
        List of split IDs
    """
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]

    all_collections = db.list_collection_names()
    split_collections = [c for c in all_collections
                        if c.startswith(collection_prefix) and c.endswith(collection_suffix)]

    # Extract split IDs
    split_ids = []
    for coll in split_collections:
        # Remove prefix and suffix to get ID
        id_str = coll[len(collection_prefix):-len(collection_suffix)]
        try:
            split_id = int(id_str)
            split_ids.append(split_id)
        except ValueError:
            continue

    client.close()
    return sorted(split_ids)


def ensure_indexes(mongo_uri: str, db_name: str, split_ids: list):
    """
    Ensure timestamp indexes exist on all split collections.

    Follows the pattern from stages 03-14 for efficient queries.
    """
    logger('', "INFO")
    logger('Ensuring timestamp indexes on all split collections...', "INFO")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]

    for split_id in split_ids:
        collection_name = f"{COLLECTION_PREFIX}{split_id}{COLLECTION_SUFFIX}"

        if collection_name in db.list_collection_names():
            collection = db[collection_name]

            # Check if index exists
            existing_indexes = list(collection.list_indexes())
            has_timestamp_index = any('timestamp' in idx.get('key', {}) for idx in existing_indexes)

            if not has_timestamp_index:
                logger(f'  Creating index on {collection_name}...', "INFO")
                collection.create_index([("timestamp", ASCENDING)], background=False)
            else:
                logger(f'  Index already exists on {collection_name}', "INFO")

    client.close()
    logger(f'Timestamp indexes verified on {len(split_ids)} collections', "INFO")


def run_episode(
    agent,
    episode,
    state_buffer: StateBuffer,
    trajectory_buffer: TrajectoryBuffer,
    agent_state: AgentState,
    reward_config: RewardConfig,
    model_config: ModelConfig,
    experiment_type: ExperimentType,
    device: str,
    deterministic: bool = False
):
    """
    Run one episode and collect trajectories.

    Args:
        agent: PPO agent model (ActorCriticTransformer, ActorCriticFeatures, or ActorCriticCodebook)
        episode: Episode object with samples
        state_buffer: Rolling window state buffer
        trajectory_buffer: Trajectory buffer for PPO updates
        agent_state: Agent trading state
        reward_config: Reward function config
        model_config: Model config
        experiment_type: Type of experiment (determines which inputs to use)
        device: Device
        deterministic: Use deterministic actions (for validation)

    Returns:
        Episode metrics dictionary
    """
    state_buffer.reset()
    agent_state.reset()

    # Get valid timesteps for this episode
    valid_steps = get_valid_timesteps(episode, model_config.window_size, model_config.horizon)

    if len(valid_steps) == 0:
        # Episode too short
        return None

    episode_returns = []

    # Fill initial buffer
    for t in range(model_config.window_size):
        sample = episode.samples[t]
        state_buffer.add(
            sample['codebook'],
            sample['features'],
            sample['timestamp']
        )

    # Run through valid timesteps
    for t in valid_steps:
        # Get current state
        state = state_buffer.get_state()
        if state is None:
            continue

        codebooks = state['codebooks'].to(device)
        features = state['features'].to(device)
        timestamps = state['timestamps'].to(device)

        # Get current sample
        current_sample = episode.samples[t]

        # Agent selects action based on experiment type
        with torch.no_grad():
            if experiment_type == ExperimentType.EXP1_BOTH_ORIGINAL:
                # Experiment 1: Both codebook + features
                action, log_prob, value = agent.act(
                    codebooks, features, timestamps,
                    deterministic=deterministic
                )
            elif experiment_type == ExperimentType.EXP2_FEATURES_ORIGINAL:
                # Experiment 2: Features only
                action, log_prob, value = agent.act(
                    features, timestamps,
                    deterministic=deterministic
                )
            else:  # EXP3_CODEBOOK_ORIGINAL
                # Experiment 3: Codebook only
                action, log_prob, value = agent.act(
                    codebooks, timestamps,
                    deterministic=deterministic
                )

        action_val = action.item()
        log_prob_val = log_prob.item() if not deterministic else 0.0
        value_val = value.item()

        # Get immediate target (one-step forward return)
        target = current_sample['target']

        # Extract volatility from features (standardized/normalized, already computed)
        # Volatility is at index 6 (feature_names[6] = "volatility")
        volatility = current_sample['features'][6].item()

        # Scale action to position using volatility
        # C=0.05 gives ~6% positions at typical volatility (σ≈0.8)
        # epsilon=0.1 floors volatility to prevent position explosions when σ→0
        position_curr = compute_volatility_scaled_position(
            action_val, volatility, vol_constant=0.05, epsilon=0.1
        )

        # Get previous position (volatility-scaled from previous timestep)
        position_prev = agent_state.current_position

        # Compute reward using simple PnL-based formula
        reward, gross_pnl, tc = compute_simple_reward(
            position_prev, position_curr, target, taker_fee=0.0005
        )

        # Unrealized PnL for next timestep
        unrealized = compute_unrealized_pnl(position_curr, target)

        # Get backward return for realized PnL tracking
        # For first valid step, use 0; otherwise use previous sample's target
        if t == valid_steps[0]:
            log_return = 0.0
        else:
            log_return = episode.samples[t - 1]['target']

        # Update agent state with scaled position
        agent_state.update(position_curr, log_return, tc, reward, unrealized, gross_pnl, volatility)
        episode_returns.append(reward)

        # Check if episode should end
        done = (t >= valid_steps[-1])

        # Store transition
        if not deterministic:
            transition = Transition(
                codebooks=codebooks.cpu(),
                features=features.cpu(),
                timestamps=timestamps.cpu(),
                action=action_val,
                log_prob=log_prob_val,
                reward=reward,
                value=value_val,
                done=done
            )
            trajectory_buffer.add(transition)

        # Add next sample to buffer
        if t + 1 < len(episode.samples):
            next_sample = episode.samples[t + 1]
            state_buffer.add(
                next_sample['codebook'],
                next_sample['features'],
                next_sample['timestamp']
            )

        if done:
            break

    # Compute episode metrics
    metrics = agent_state.get_metrics()
    metrics['episode_length'] = len(valid_steps)

    return metrics


def train_epoch(
    agent,
    optimizer: torch.optim.Optimizer,
    episodes: list,
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    model_config: ModelConfig,
    experiment_type: ExperimentType,
    device: str
):
    """
    Train one epoch across multiple episodes.

    Returns:
        Training metrics dictionary
    """
    agent.train()

    state_buffer = StateBuffer(model_config.window_size)
    trajectory_buffer = TrajectoryBuffer(ppo_config.buffer_capacity)
    agent_state = AgentState()

    epoch_metrics = {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'episode_count': 0,
        'avg_episode_length': 0.0,
        'total_policy_loss': 0.0,
        'total_value_loss': 0.0,
        'total_entropy': 0.0,
        'n_ppo_updates': 0
    }

    episode_returns = []

    # Use ALL training episodes each epoch
    total_episodes = len(episodes)

    # Group episodes by parent_id for aggregated logging
    from collections import defaultdict
    day_metrics = defaultdict(lambda: {
        'total_reward': 0.0, 'total_pnl': 0.0, 'chunks': 0,
        'total_trades': 0, 'total_steps': 0,
        'sum_mean_pos': 0.0, 'max_pos': 0.0,
        'sum_mean_vol': 0.0, 'min_vol': float('inf'), 'max_vol': float('-inf'),
        'sum_gross_pnl_per_trade': 0.0, 'sum_tc_per_trade': 0.0
    })

    for ep_idx, episode in enumerate(episodes, 1):
        metrics = run_episode(
            agent, episode, state_buffer, trajectory_buffer, agent_state,
            reward_config, model_config, experiment_type, device, deterministic=False
        )

        if metrics is None:
            continue

        epoch_metrics['total_reward'] += metrics['total_reward']
        epoch_metrics['total_pnl'] += metrics['total_pnl']
        epoch_metrics['episode_count'] += 1
        epoch_metrics['avg_episode_length'] += metrics['episode_length']
        episode_returns.append(metrics['total_reward'])

        # Aggregate metrics by parent episode (day)
        parent_id = episode.parent_id if episode.parent_id is not None else ep_idx
        dm = day_metrics[parent_id]
        dm['total_reward'] += metrics['total_reward']
        dm['total_pnl'] += metrics['total_pnl']
        dm['chunks'] += 1
        dm['total_trades'] += metrics['trade_count']
        dm['total_steps'] += metrics['episode_length']
        dm['sum_mean_pos'] += metrics['mean_abs_position']
        dm['max_pos'] = max(dm['max_pos'], metrics['max_position'])
        dm['sum_mean_vol'] += metrics['mean_volatility']
        dm['min_vol'] = min(dm['min_vol'], metrics['min_volatility'])
        dm['max_vol'] = max(dm['max_vol'], metrics['max_volatility'])
        dm['sum_gross_pnl_per_trade'] += metrics['avg_gross_pnl_per_trade']
        dm['sum_tc_per_trade'] += metrics['avg_tc_per_trade']

        # Log when completing a parent episode (all chunks done)
        is_last_chunk = (ep_idx == total_episodes or
                        (ep_idx < total_episodes and episodes[ep_idx].parent_id != parent_id))

        if is_last_chunk:
            from src.utils.logging import logger
            n_chunks = dm['chunks']
            logger(f'    Day {len([k for k in day_metrics.keys() if day_metrics[k]["chunks"] > 0])}/{len(set(e.parent_id if e.parent_id is not None else i for i, e in enumerate(episodes, 1)))} '
                   f'({n_chunks} chunks) - '
                   f'Reward: {dm["total_reward"]:.4f}, '
                   f'PnL: {dm["total_pnl"]:.4f}, '
                   f'Trades: {dm["total_trades"]} ({dm["total_trades"]/dm["total_steps"]:.2%}), '
                   f'Pos[μ={dm["sum_mean_pos"]/n_chunks:.4f}, max={dm["max_pos"]:.4f}], '
                   f'Vol[μ={dm["sum_mean_vol"]/n_chunks:.4f}, range=[{dm["min_vol"]:.4f},{dm["max_vol"]:.4f}]], '
                   f'Gross PnL/Trade: {dm["sum_gross_pnl_per_trade"]/n_chunks:.6f}, '
                   f'TC/Trade: {dm["sum_tc_per_trade"]/n_chunks:.6f}, '
                   f'Steps: {dm["total_steps"]}', "INFO")

        # Perform PPO update if buffer is full
        if trajectory_buffer.is_full():
            loss_metrics = ppo_update(
                agent, trajectory_buffer, optimizer, ppo_config, experiment_type, device
            )
            epoch_metrics['total_policy_loss'] += loss_metrics['policy_loss']
            epoch_metrics['total_value_loss'] += loss_metrics['value_loss']
            epoch_metrics['total_entropy'] += loss_metrics['entropy']
            epoch_metrics['n_ppo_updates'] += 1
            trajectory_buffer.clear()

    # Final PPO update with remaining trajectories
    if len(trajectory_buffer) > 0:
        loss_metrics = ppo_update(
            agent, trajectory_buffer, optimizer, ppo_config, experiment_type, device
        )
        epoch_metrics['total_policy_loss'] += loss_metrics['policy_loss']
        epoch_metrics['total_value_loss'] += loss_metrics['value_loss']
        epoch_metrics['total_entropy'] += loss_metrics['entropy']
        epoch_metrics['n_ppo_updates'] += 1
        trajectory_buffer.clear()

    # Compute averages
    if epoch_metrics['episode_count'] > 0:
        epoch_metrics['avg_reward'] = epoch_metrics['total_reward'] / epoch_metrics['episode_count']
        epoch_metrics['avg_pnl'] = epoch_metrics['total_pnl'] / epoch_metrics['episode_count']
        epoch_metrics['avg_episode_length'] /= epoch_metrics['episode_count']
        epoch_metrics['sharpe'] = compute_sharpe_ratio(episode_returns)
    else:
        epoch_metrics['avg_reward'] = 0.0
        epoch_metrics['avg_pnl'] = 0.0
        epoch_metrics['avg_episode_length'] = 0.0
        epoch_metrics['sharpe'] = 0.0

    # Compute loss averages
    if epoch_metrics['n_ppo_updates'] > 0:
        epoch_metrics['avg_policy_loss'] = epoch_metrics['total_policy_loss'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_value_loss'] = epoch_metrics['total_value_loss'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_entropy'] = epoch_metrics['total_entropy'] / epoch_metrics['n_ppo_updates']
    else:
        epoch_metrics['avg_policy_loss'] = 0.0
        epoch_metrics['avg_value_loss'] = 0.0
        epoch_metrics['avg_entropy'] = 0.0

    return epoch_metrics


def validate_epoch(
    agent,
    episodes: list,
    reward_config: RewardConfig,
    model_config: ModelConfig,
    experiment_type: ExperimentType,
    device: str
):
    """
    Validate on validation episodes.

    Returns:
        Validation metrics dictionary
    """
    agent.eval()

    state_buffer = StateBuffer(model_config.window_size)
    trajectory_buffer = TrajectoryBuffer(1000)  # Not used, just placeholder
    agent_state = AgentState()

    val_metrics = {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'episode_count': 0
    }

    episode_returns = []
    total_episodes = len(episodes)

    # Group episodes by parent_id for aggregated logging
    from collections import defaultdict
    day_metrics = defaultdict(lambda: {
        'total_reward': 0.0, 'total_pnl': 0.0, 'chunks': 0,
        'total_trades': 0, 'total_steps': 0,
        'sum_mean_pos': 0.0, 'max_pos': 0.0,
        'sum_mean_vol': 0.0, 'min_vol': float('inf'), 'max_vol': float('-inf'),
        'sum_gross_pnl_per_trade': 0.0, 'sum_tc_per_trade': 0.0
    })

    for ep_idx, episode in enumerate(episodes, 1):
        metrics = run_episode(
            agent, episode, state_buffer, trajectory_buffer, agent_state,
            reward_config, model_config, experiment_type, device, deterministic=True
        )

        if metrics is None:
            continue

        val_metrics['total_reward'] += metrics['total_reward']
        val_metrics['total_pnl'] += metrics['total_pnl']
        val_metrics['episode_count'] += 1
        episode_returns.append(metrics['total_reward'])

        # Aggregate metrics by parent episode (day)
        parent_id = episode.parent_id if episode.parent_id is not None else ep_idx
        dm = day_metrics[parent_id]
        dm['total_reward'] += metrics['total_reward']
        dm['total_pnl'] += metrics['total_pnl']
        dm['chunks'] += 1
        dm['total_trades'] += metrics['trade_count']
        dm['total_steps'] += metrics['episode_length']
        dm['sum_mean_pos'] += metrics['mean_abs_position']
        dm['max_pos'] = max(dm['max_pos'], metrics['max_position'])
        dm['sum_mean_vol'] += metrics['mean_volatility']
        dm['min_vol'] = min(dm['min_vol'], metrics['min_volatility'])
        dm['max_vol'] = max(dm['max_vol'], metrics['max_volatility'])
        dm['sum_gross_pnl_per_trade'] += metrics['avg_gross_pnl_per_trade']
        dm['sum_tc_per_trade'] += metrics['avg_tc_per_trade']

        # Log when completing a parent episode (all chunks done)
        is_last_chunk = (ep_idx == total_episodes or
                        (ep_idx < total_episodes and episodes[ep_idx].parent_id != parent_id))

        if is_last_chunk:
            from src.utils.logging import logger
            n_chunks = dm['chunks']
            logger(f'    Day {len([k for k in day_metrics.keys() if day_metrics[k]["chunks"] > 0])}/{len(set(e.parent_id if e.parent_id is not None else i for i, e in enumerate(episodes, 1)))} '
                   f'({n_chunks} chunks) - '
                   f'Reward: {dm["total_reward"]:.4f}, '
                   f'PnL: {dm["total_pnl"]:.4f}, '
                   f'Trades: {dm["total_trades"]} ({dm["total_trades"]/dm["total_steps"]:.2%}), '
                   f'Pos[μ={dm["sum_mean_pos"]/n_chunks:.4f}, max={dm["max_pos"]:.4f}], '
                   f'Vol[μ={dm["sum_mean_vol"]/n_chunks:.4f}, range=[{dm["min_vol"]:.4f},{dm["max_vol"]:.4f}]], '
                   f'Gross PnL/Trade: {dm["sum_gross_pnl_per_trade"]/n_chunks:.6f}, '
                   f'TC/Trade: {dm["sum_tc_per_trade"]/n_chunks:.6f}, '
                   f'Steps: {dm["total_steps"]}', "INFO")

    # Compute averages
    if val_metrics['episode_count'] > 0:
        val_metrics['avg_reward'] = val_metrics['total_reward'] / val_metrics['episode_count']
        val_metrics['avg_pnl'] = val_metrics['total_pnl'] / val_metrics['episode_count']
        val_metrics['sharpe'] = compute_sharpe_ratio(episode_returns)
    else:
        val_metrics['avg_reward'] = 0.0
        val_metrics['avg_pnl'] = 0.0
        val_metrics['sharpe'] = 0.0

    return val_metrics


def train_split(
    split_id: int,
    config: ExperimentConfig,
    device: torch.device
):
    """
    Train PPO agent on one split.

    Returns:
        Dictionary with training results
    """
    logger('', "INFO")
    logger(f'Training agent on split {split_id}...', "INFO")

    # Get experiment type from config
    experiment_type = config.data.experiment_type

    # Initialize episode loader with experiment type and chunk size
    data_config = config.data
    data_config.split_ids = [split_id]

    episode_loader = EpisodeLoader(data_config, episode_chunk_size=EPISODE_CHUNK_SIZE)

    # Load episodes
    logger('Loading training episodes...', "INFO")
    train_episodes = episode_loader.load_episodes(split_id, role='train')
    logger(f'  Loaded {len(train_episodes)} training episodes', "INFO")

    logger('Loading validation episodes...', "INFO")
    val_episodes = episode_loader.load_episodes(split_id, role='val')
    logger(f'  Loaded {len(val_episodes)} validation episodes', "INFO")

    # Initialize agent based on experiment type
    logger(f'Initializing agent for Experiment {experiment_type.value}...', "INFO")
    if experiment_type == ExperimentType.EXP1_BOTH_ORIGINAL:
        agent = ActorCriticTransformer(config.model).to(device)
    elif experiment_type == ExperimentType.EXP2_FEATURES_ORIGINAL:
        agent = ActorCriticFeatures(config.model).to(device)
    else:  # EXP3_CODEBOOK_ORIGINAL
        agent = ActorCriticCodebook(config.model).to(device)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=config.ppo.learning_rate,
        weight_decay=config.ppo.weight_decay
    )

    # Learning rate scheduler - reduces LR when validation Sharpe plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )

    logger(f'Agent initialized: {agent.count_parameters():,} parameters', "INFO")
    logger(f'Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=2)', "INFO")

    # Training loop
    metrics_logger = MetricsLogger(log_dir=str(LOG_DIR))
    best_val_sharpe = float('-inf')
    patience_counter = 0

    for epoch in range(config.training.max_epochs):
        epoch_start = time.time()

        logger('', "INFO")
        logger(f'Epoch {epoch + 1}/{config.training.max_epochs}', "INFO")

        # Training
        train_metrics = train_epoch(
            agent, optimizer, train_episodes,
            config.ppo, config.reward, config.model, experiment_type, device
        )

        logger(f'  Train - Sharpe: {train_metrics["sharpe"]:.4f}, '
               f'Avg Reward: {train_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {train_metrics["avg_pnl"]:.4f}', "INFO")
        logger(f'  Losses - Policy: {train_metrics["avg_policy_loss"]:.4f}, '
               f'Value: {train_metrics["avg_value_loss"]:.4f}, '
               f'Entropy: {train_metrics["avg_entropy"]:.4f}', "INFO")

        # Validation (every epoch)
        val_metrics = validate_epoch(
            agent, val_episodes, config.reward, config.model, experiment_type, device
        )

        logger(f'  Val - Sharpe: {val_metrics["sharpe"]:.4f}, '
               f'Avg Reward: {val_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {val_metrics["avg_pnl"]:.4f}', "INFO")

        # Update learning rate based on validation Sharpe
        scheduler.step(val_metrics["sharpe"])
        current_lr = optimizer.param_groups[0]['lr']
        logger(f'  Learning rate: {current_lr:.6f}', "INFO")

        # Log to MLflow
        mlflow.log_metric("train_sharpe", train_metrics["sharpe"], step=epoch)
        mlflow.log_metric("train_avg_reward", train_metrics["avg_reward"], step=epoch)
        mlflow.log_metric("train_avg_pnl", train_metrics["avg_pnl"], step=epoch)
        mlflow.log_metric("train_policy_loss", train_metrics["avg_policy_loss"], step=epoch)
        mlflow.log_metric("train_value_loss", train_metrics["avg_value_loss"], step=epoch)
        mlflow.log_metric("train_entropy", train_metrics["avg_entropy"], step=epoch)
        mlflow.log_metric("val_sharpe", val_metrics["sharpe"], step=epoch)
        mlflow.log_metric("val_avg_reward", val_metrics["avg_reward"], step=epoch)
        mlflow.log_metric("val_avg_pnl", val_metrics["avg_pnl"], step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Save checkpoint if best
        if val_metrics["sharpe"] > best_val_sharpe:
            best_val_sharpe = val_metrics["sharpe"]
            patience_counter = 0

            save_checkpoint(
                agent, optimizer, epoch, split_id, val_metrics["sharpe"],
                checkpoint_dir=str(CHECKPOINT_DIR)
            )
            logger(f'  ✓ New best model saved (Sharpe: {best_val_sharpe:.4f})', "INFO")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.patience:
            logger(f'  Early stopping triggered (patience: {config.training.patience})', "INFO")
            break

        epoch_time = time.time() - epoch_start
        logger(f'  Epoch time: {epoch_time:.2f}s', "INFO")

    episode_loader.close()

    return {
        'best_val_sharpe': best_val_sharpe,
        'epochs_trained': epoch + 1
    }


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PPO Agent Training with Different Experiments')
    parser.add_argument('--experiment', type=int, default=None, choices=[1, 2, 3],
                        help='Experiment type: 1=Both sources (original), 2=Features only (original), '
                             '3=Codebook only (original). '
                             'If not specified, runs all 3 experiments sequentially.')
    parser.add_argument('--splits', type=str, default=None,
                        help='Comma-separated list of split IDs to train (e.g., "0,1,2"). '
                             'If not specified, trains on all available splits.')
    parser.add_argument('--max-splits', type=int, default=None,
                        help='Maximum number of splits to train on (uses first N splits). '
                             'Useful for quick testing.')
    args = parser.parse_args()

    # Determine which experiments to run
    if args.experiment is not None:
        experiments_to_run = [args.experiment]
    else:
        # Run all 3 experiments by default
        experiments_to_run = [1, 2, 3]

    # Map experiment number to enum
    experiment_mapping = {
        1: ExperimentType.EXP1_BOTH_ORIGINAL,
        2: ExperimentType.EXP2_FEATURES_ORIGINAL,
        3: ExperimentType.EXP3_CODEBOOK_ORIGINAL
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger('', "INFO")
    logger(f'Device: {device}', "INFO")

    if device.type == 'cuda':
        logger(f'CUDA Device: {torch.cuda.get_device_name(0)}', "INFO")
        logger(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB', "INFO")

    # Discover splits
    logger('', "INFO")
    logger('Discovering available splits...', "INFO")
    split_ids = discover_splits(MONGO_URI, DB_NAME, COLLECTION_PREFIX, COLLECTION_SUFFIX)

    if not split_ids:
        raise ValueError(f"No splits found in database '{DB_NAME}'")

    logger(f'Found {len(split_ids)} splits: {split_ids}', "INFO")

    # Filter splits based on command-line arguments
    if args.splits is not None:
        # User specified exact splits to run
        requested_splits = [int(s.strip()) for s in args.splits.split(',')]
        split_ids = [s for s in split_ids if s in requested_splits]
        logger(f'Using user-specified splits: {split_ids}', "INFO")
    elif args.max_splits is not None:
        # Limit to first N splits
        split_ids = split_ids[:args.max_splits]
        logger(f'Limiting to first {args.max_splits} splits: {split_ids}', "INFO")

    if not split_ids:
        raise ValueError("No splits selected after filtering")

    # Ensure indexes
    ensure_indexes(MONGO_URI, DB_NAME, split_ids)

    # Setup MLflow
    logger('', "INFO")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}', "INFO")
    logger(f'MLflow experiment: {MLFLOW_EXPERIMENT_NAME}', "INFO")

    # Create artifact directories
    ARTIFACT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Run each experiment
    for exp_num in experiments_to_run:
        selected_experiment = experiment_mapping[exp_num]

        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger(f'PPO AGENT TRAINING - EXPERIMENT {exp_num} (STAGE 18)', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Experiment: {selected_experiment.name}', "INFO")

        # Create experiment config with selected experiment type
        config = ExperimentConfig(
            name=f"ppo_exp{exp_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model=ModelConfig(window_size=WINDOW_SIZE, horizon=HORIZON),
            ppo=PPOConfig(),
            reward=RewardConfig(),
            training=TrainingConfig(device=str(device)),
            data=DataConfig(
                database_name=DB_NAME,
                split_ids=split_ids,
                experiment_type=selected_experiment
            )
        )

        # Save config
        exp_artifact_dir = ARTIFACT_BASE_DIR / f"experiment_{exp_num}"
        exp_artifact_dir.mkdir(parents=True, exist_ok=True)
        config_path = exp_artifact_dir / "experiment_config.json"
        config.save(str(config_path))
        logger(f'Experiment config saved to: {config_path}', "INFO")

        # Log training configuration
        logger('', "INFO")
        logger('Training Configuration:', "INFO")
        logger(f'  Max epochs: {config.training.max_epochs}', "INFO")
        logger(f'  Episodes per epoch: ALL (no limit)', "INFO")
        logger(f'  Splits to train: {len(split_ids)}', "INFO")
        logger(f'  Early stopping patience: {config.training.patience} epochs', "INFO")

        # Estimate training time (assuming ~40 train episodes per split, ~30s per episode)
        # Each epoch processes ALL training episodes
        # Early stopping (patience=3) will likely stop around epoch 5-7
        estimated_episodes_per_split = 40  # typical 80% of ~50 total episodes
        estimated_time_per_epoch = (estimated_episodes_per_split * 30) / 3600  # hours
        estimated_time_per_split_max = config.training.max_epochs * estimated_time_per_epoch
        estimated_time_per_split_typical = 5 * estimated_time_per_epoch  # with early stopping
        total_estimated_time_max = estimated_time_per_split_max * len(split_ids)
        total_estimated_time_typical = estimated_time_per_split_typical * len(split_ids)
        logger(f'  Estimated time per split: ~{estimated_time_per_split_typical:.1f}h (typical) to {estimated_time_per_split_max:.1f}h (max)', "INFO")
        logger(f'  Total estimated time: ~{total_estimated_time_typical:.1f}h (typical) to {total_estimated_time_max:.1f}h (max)', "INFO")

        # Main training loop for this experiment
        with mlflow.start_run(run_name=config.name):
            # Log configuration
            mlflow.log_params(config.to_dict())
            mlflow.log_artifact(str(config_path))

            # Train on each split
            all_results = {}

            for split_id in split_ids:
                logger('', "INFO")
                logger('=' * 100, "INFO")
                logger(f'SPLIT {split_id}', "INFO")
                logger('=' * 100, "INFO")

                with mlflow.start_run(run_name=f"split_{split_id}", nested=True):
                    mlflow.log_param("split_id", split_id)
                    mlflow.log_param("experiment_type", exp_num)

                    results = train_split(split_id, config, device)
                    all_results[split_id] = results

                    mlflow.log_metric("best_val_sharpe", results['best_val_sharpe'])
                    mlflow.log_metric("epochs_trained", results['epochs_trained'])

                    logger('', "INFO")
                    logger(f'Split {split_id} complete:', "INFO")
                    logger(f'  Best validation Sharpe: {results["best_val_sharpe"]:.4f}', "INFO")
                    logger(f'  Epochs trained: {results["epochs_trained"]}', "INFO")

            # Summary for this experiment
            logger('', "INFO")
            logger('=' * 100, "INFO")
            logger(f'EXPERIMENT {exp_num} COMPLETE', "INFO")
            logger('=' * 100, "INFO")

            avg_sharpe = np.mean([r['best_val_sharpe'] for r in all_results.values()])
            logger(f'Average validation Sharpe across splits: {avg_sharpe:.4f}', "INFO")
            logger(f'Checkpoints saved to: {CHECKPOINT_DIR}', "INFO")

            mlflow.log_metric("avg_val_sharpe_across_splits", avg_sharpe)

    # Final summary
    logger('', "INFO")
    logger('=' * 100, "INFO")
    logger('ALL EXPERIMENTS COMPLETE', "INFO")
    logger('=' * 100, "INFO")
    logger(f'Completed {len(experiments_to_run)} experiment(s)', "INFO")
    logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")


if __name__ == "__main__":
    # Check if running from orchestrator
    is_orchestrated = os.environ.get('PIPELINE_ORCHESTRATED', 'false') == 'true'

    start_time = time.time()

    try:
        main()

        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logger('', "INFO")
        logger(f'Total execution time: {hours}h {minutes}m {seconds}s', "INFO")
        logger('Stage 17 completed successfully', "INFO")

    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
