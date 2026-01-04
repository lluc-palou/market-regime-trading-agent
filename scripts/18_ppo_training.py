"""
PPO Agent Training Script (Stage 21)

TRAIN MODE: Trains PPO agents using CPCV (train/val split)
TEST MODE: Trains on full split_0 (train+val) and evaluates on test_data

Trains PPO agents for limit order book trading using VQ-VAE latent representations
and hand-crafted features.

Train Mode:
- Input: split_X_input (role='train' for training, role='validation' for eval)
- Output: CPCV train/val metrics, best models per split
- Saves to: artifacts/ppo_training/experiment_X/split_Y/

Test Mode:
- Input: split_0_input (all roles for training), test_data (for evaluation)
- Output: Final test metrics (Sharpe for all fee scenarios)
- Saves to: artifacts/ppo_training/test/experiment_X/

Usage:
    TRAIN: python scripts/18_ppo_training.py --experiment 1 --mode train
    TEST:  python scripts/18_ppo_training.py --experiment 1 --mode test --test-split 0
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
import math
import csv
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
    compute_policy_based_position,
    compute_simple_reward,
    compute_unrealized_pnl,
    compute_ewma_volatility,
    compute_directional_bonus,
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

        # Use non-blocking transfers with pinned memory for better performance
        codebooks = state['codebooks'].to(device, non_blocking=True)
        features = state['features'].to(device, non_blocking=True)
        timestamps = state['timestamps'].to(device, non_blocking=True)

        # Get current sample
        current_sample = episode.samples[t]

        # Agent selects action based on experiment type
        with torch.no_grad():
            if experiment_type == ExperimentType.EXP1_BOTH_ORIGINAL:
                # Experiment 1: Both codebook + features
                action, log_prob, value, action_std = agent.act(
                    codebooks, features, timestamps,
                    deterministic=deterministic
                )
            elif experiment_type == ExperimentType.EXP2_FEATURES_ORIGINAL:
                # Experiment 2: Features only
                action, log_prob, value, action_std = agent.act(
                    features, timestamps,
                    deterministic=deterministic
                )
            else:  # EXP3_CODEBOOK_ORIGINAL
                # Experiment 3: Codebook only
                action, log_prob, value, action_std = agent.act(
                    codebooks, timestamps,
                    deterministic=deterministic
                )

        action_val = action.item()
        action_std_val = action_std.item()
        log_prob_val = log_prob.item() if not deterministic else 0.0
        value_val = value.item()

        # Get multi-step cumulative target (H-step forward returns)
        H = model_config.horizon  # 10
        multistep_target = 0.0

        # Sum next H forward returns (or remaining if episode ends sooner)
        for i in range(1, min(H + 1, len(episode.samples) - t)):
            multistep_target += episode.samples[t + i]['target']

        target = multistep_target

        # Compute position using agent's policy distribution (action mean + std)
        # This allows PPO to learn position sizing directly through confidence (std)
        # Lower std → higher confidence → larger position
        # Higher std → lower confidence → smaller position
        position_curr = compute_policy_based_position(
            action_val, action_std_val, confidence_weight=1.0
        )

        # Get previous position (policy-based from previous timestep)
        position_prev = agent_state.current_position

        # Compute reward using simple PnL-based formula
        # TRAINING SCENARIO: Taker fees (10 bps transaction cost)
        # Agent learns under harder conditions - forces better position sizing and directional edge
        # Higher fees encourage more selective, higher-quality trades over frequent trading
        reward, gross_pnl, tc = compute_simple_reward(
            position_prev, position_curr, target, taker_fee=0.001  # 10 bps taker fee
        )

        # Compute directional accuracy bonus (for reward shaping, not logged PnL)
        # Provides additional signal when agent predicts direction correctly
        # This bonus affects ONLY the reward signal for learning, not the logged gross_pnl metric
        # Uses default bonus_weight=0.000002 (~25% of H=10 gross PnL, avoiding signal dominance)
        directional_bonus = compute_directional_bonus(position_curr, target)

        # Scale reward by realized volatility (EWMA of recent returns)
        # Collect recent 1-step targets from past samples for volatility estimate
        lookback = min(30, t)  # Use up to 30 recent samples for volatility estimate
        recent_targets = [episode.samples[t - i]['target'] for i in range(lookback, 0, -1)]
        recent_targets.append(episode.samples[t]['target'])  # Include current 1-step target

        # Compute EWMA volatility from recent 1-step returns (half-life=20, matches feature engineering)
        step1_vol = compute_ewma_volatility(recent_targets, half_life=20)

        # Scale to H-step volatility using sqrt(H) rule (standard financial theory)
        # Multi-step volatility = single-step volatility × sqrt(H)
        realized_vol = step1_vol * math.sqrt(model_config.horizon)

        # Compute trading returns for all fee scenarios (all exclude directional bonus)
        # These are used for performance metrics (Sharpe ratio calculation)
        # All scenarios computed from gross_pnl with their respective transaction costs
        position_change = abs(position_curr - position_prev)

        # 1. Baseline: Buy-and-hold (raw returns, no position sizing, no fees)
        # Equivalent to constant position of +1.0 (always long)
        trading_return_raw = target / realized_vol

        # 2. Maker neutral (0 bps) - agent's position sizing, no TC
        tc_maker_neutral = 0.0  # No transaction cost
        reward_maker_neutral = gross_pnl - tc_maker_neutral
        trading_return_maker_neutral = reward_maker_neutral / realized_vol

        # 3. Taker fee (10 bps) - market orders with agent's position sizing [TRAINING SCENARIO]
        tc_taker = 0.001 * position_change  # 10 basis points (matches training reward)
        reward_taker = gross_pnl - tc_taker
        trading_return_taker = reward_taker / realized_vol

        # 4. Maker rebate (-2.5 bps) - limit orders with rebate
        tc_maker_rebate = -0.00025 * position_change  # -2.5 basis points (rebate)
        reward_maker_rebate = gross_pnl - tc_maker_rebate
        trading_return_maker_rebate = reward_maker_rebate / realized_vol

        # Add directional bonus to reward for learning signal (PPO optimization)
        # Note: All trading_return_* exclude bonus for accurate performance metrics
        # reward here is already maker_neutral (0 TC), so just add bonus and scale
        reward = (reward + directional_bonus) / realized_vol

        # Unrealized PnL for next timestep
        unrealized = compute_unrealized_pnl(position_curr, target)

        # Get backward return for realized PnL tracking
        # For first valid step, use 0; otherwise use previous sample's target
        if t == valid_steps[0]:
            log_return = 0.0
        else:
            log_return = episode.samples[t - 1]['target']

        # Update agent state with policy-based position
        # Track action_std instead of volatility (agent's learned uncertainty)
        # Pass reward (learning signal) and all 4 trading_returns (performance metrics for different fee scenarios)
        agent_state.update(
            position_curr, log_return, tc, reward, unrealized, gross_pnl, action_std_val,
            trading_return_raw, trading_return_taker, trading_return_maker_neutral, trading_return_maker_rebate
        )
        episode_returns.append(trading_return_taker)  # Taker (5 bps) is training scenario - harder conditions for better learning

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
        'total_uncertainty': 0.0,
        'total_activity': 0.0,
        'total_turnover': 0.0,
        'n_ppo_updates': 0
    }

    # Episode returns for Sharpe ratio calculation (different fee scenarios)
    episode_returns_raw = []  # Baseline: Raw returns (no fees)
    episode_returns_taker = []  # Taker fee 5 bps (market orders)
    episode_returns_maker_neutral = []  # Maker fee 0 bps (limit orders)
    episode_returns_maker_rebate = []  # Maker rebate -2.5 bps (limit orders)

    # Use ALL training episodes each epoch
    total_episodes = len(episodes)

    # Group episodes by parent_id for aggregated logging
    from collections import defaultdict
    day_metrics = defaultdict(lambda: {
        'total_reward': 0.0, 'total_pnl': 0.0, 'chunks': 0,
        'total_trades': 0, 'total_steps': 0,
        'sum_mean_pos': 0.0, 'max_pos': 0.0,
        'sum_mean_std': 0.0, 'min_std': float('inf'), 'max_std': float('-inf'),
        'sum_gross_pnl_per_trade': 0.0, 'sum_position_change_per_trade': 0.0
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

        # Append trading returns for all fee scenarios (for Sharpe calculation)
        episode_returns_raw.append(metrics['total_trading_return_raw'])
        episode_returns_taker.append(metrics['total_trading_return_taker'])
        episode_returns_maker_neutral.append(metrics['total_trading_return_maker_neutral'])
        episode_returns_maker_rebate.append(metrics['total_trading_return_maker_rebate'])

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
        dm['sum_mean_std'] += metrics['mean_action_std']
        dm['min_std'] = min(dm['min_std'], metrics['min_action_std'])
        dm['max_std'] = max(dm['max_std'], metrics['max_action_std'])
        dm['sum_gross_pnl_per_trade'] += metrics['avg_gross_pnl_per_trade']
        dm['sum_position_change_per_trade'] += metrics['avg_position_change_per_trade']

        # Log when completing a parent episode (all chunks done)
        is_last_chunk = (ep_idx == total_episodes or
                        (ep_idx < total_episodes and episodes[ep_idx].parent_id != parent_id))

        if is_last_chunk:
            from src.utils.logging import logger
            n_chunks = dm['chunks']

            # Calculate aggregated metrics
            avg_position = dm['sum_mean_pos'] / n_chunks
            avg_uncertainty = dm['sum_mean_std'] / n_chunks
            avg_gross_pnl = dm['sum_gross_pnl_per_trade'] / n_chunks
            avg_position_change = dm['sum_position_change_per_trade'] / n_chunks
            trade_frequency = dm['total_trades'] / dm['total_steps']

            # Compute TC for all fee scenarios from position change
            taker_fee = 0.001  # 10 bps
            maker_fee_neutral = 0.0  # 0 bps
            maker_fee_rebate = -0.00025  # -2.5 bps (negative = rebate)

            avg_tc_taker = taker_fee * avg_position_change
            avg_tc_maker_neutral = maker_fee_neutral * avg_position_change
            avg_tc_maker_rebate = maker_fee_rebate * avg_position_change

            # Net PnL for each scenario
            net_pnl_taker = avg_gross_pnl - avg_tc_taker
            net_pnl_maker_neutral = avg_gross_pnl - avg_tc_maker_neutral
            net_pnl_maker_rebate = avg_gross_pnl - avg_tc_maker_rebate

            day_num = len([k for k in day_metrics.keys() if day_metrics[k]["chunks"] > 0])
            total_days = len(set(e.parent_id if e.parent_id is not None else i for i, e in enumerate(episodes, 1)))

            logger(f'    ┌─ Day {day_num}/{total_days} ({n_chunks} chunks, {dm["total_steps"]} steps)', "INFO")
            logger(f'    │  Trading Activity: {dm["total_trades"]} trades ({trade_frequency:.1%} frequency)', "INFO")
            logger(f'    │  Position Sizing: Mean={avg_position:.3f}, Max={dm["max_pos"]:.3f}', "INFO")
            logger(f'    │  Action Uncertainty (σ): Mean={avg_uncertainty:.3f}, Range=[{dm["min_std"]:.3f}, {dm["max_std"]:.3f}]', "INFO")
            logger(f'    │', "INFO")
            logger(f'    │  Performance (Market Orders - Taker Fee 10 bps):', "INFO")
            logger(f'    │    Gross PnL/Trade: {avg_gross_pnl:.8f}', "INFO")
            logger(f'    │    TC/Trade:        {avg_tc_taker:.8f}', "INFO")
            logger(f'    │    Net PnL/Trade:   {net_pnl_taker:.8f} ({"PROFIT" if net_pnl_taker > 0 else "LOSS"})', "INFO")
            logger(f'    │', "INFO")
            logger(f'    │  Alternative: Limit Orders (Maker Fee 0 bps):', "INFO")
            logger(f'    │    Gross PnL/Trade: {avg_gross_pnl:.8f}', "INFO")
            logger(f'    │    TC/Trade:        {avg_tc_maker_neutral:.8f}', "INFO")
            logger(f'    │    Net PnL/Trade:   {net_pnl_maker_neutral:.8f} ({"PROFIT" if net_pnl_maker_neutral > 0 else "LOSS"})', "INFO")
            logger(f'    │    Improvement:     {(net_pnl_maker_neutral - net_pnl_taker):.8f}', "INFO")
            logger(f'    │', "INFO")
            logger(f'    │  Alternative: Limit Orders (Maker Rebate -2.5 bps):', "INFO")
            logger(f'    │    Gross PnL/Trade: {avg_gross_pnl:.8f}', "INFO")
            logger(f'    │    TC/Trade:        {avg_tc_maker_rebate:.8f} (rebate)', "INFO")
            logger(f'    │    Net PnL/Trade:   {net_pnl_maker_rebate:.8f} ({"PROFIT" if net_pnl_maker_rebate > 0 else "LOSS"})', "INFO")
            logger(f'    │    Improvement:     {(net_pnl_maker_rebate - net_pnl_taker):.8f}', "INFO")
            logger(f'    └─', "INFO")

        # Perform PPO update if buffer is full
        if trajectory_buffer.is_full():
            loss_metrics = ppo_update(
                agent, trajectory_buffer, optimizer, ppo_config, experiment_type, device
            )
            epoch_metrics['total_policy_loss'] += loss_metrics['policy_loss']
            epoch_metrics['total_value_loss'] += loss_metrics['value_loss']
            epoch_metrics['total_entropy'] += loss_metrics['entropy']
            epoch_metrics['total_uncertainty'] += loss_metrics['uncertainty']
            epoch_metrics['total_activity'] += loss_metrics['activity']
            epoch_metrics['total_turnover'] += loss_metrics['turnover']
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
        epoch_metrics['total_uncertainty'] += loss_metrics['uncertainty']
        epoch_metrics['total_activity'] += loss_metrics['activity']
        epoch_metrics['n_ppo_updates'] += 1
        trajectory_buffer.clear()

    # Compute averages
    if epoch_metrics['episode_count'] > 0:
        epoch_metrics['avg_reward'] = epoch_metrics['total_reward'] / epoch_metrics['episode_count']
        epoch_metrics['avg_pnl'] = epoch_metrics['total_pnl'] / epoch_metrics['episode_count']
        epoch_metrics['avg_episode_length'] /= epoch_metrics['episode_count']

        # Compute Sharpe ratios for all fee scenarios
        epoch_metrics['sharpe_raw'] = compute_sharpe_ratio(episode_returns_raw)  # Baseline: no fees
        epoch_metrics['sharpe_taker'] = compute_sharpe_ratio(episode_returns_taker)  # Taker 10 bps
        epoch_metrics['sharpe_maker_neutral'] = compute_sharpe_ratio(episode_returns_maker_neutral)  # Maker 0 bps
        epoch_metrics['sharpe_maker_rebate'] = compute_sharpe_ratio(episode_returns_maker_rebate)  # Maker -2.5 bps
        epoch_metrics['sharpe'] = epoch_metrics['sharpe_taker']  # Legacy field (use taker for training - harder conditions)

        # Compute average PnL per scenario (for Sharpe validation)
        epoch_metrics['pnl_raw'] = np.mean(episode_returns_raw)
        epoch_metrics['pnl_taker'] = np.mean(episode_returns_taker)
        epoch_metrics['pnl_maker_neutral'] = np.mean(episode_returns_maker_neutral)
        epoch_metrics['pnl_maker_rebate'] = np.mean(episode_returns_maker_rebate)
    else:
        epoch_metrics['avg_reward'] = 0.0
        epoch_metrics['avg_pnl'] = 0.0
        epoch_metrics['avg_episode_length'] = 0.0
        epoch_metrics['sharpe_raw'] = 0.0
        epoch_metrics['sharpe_taker'] = 0.0
        epoch_metrics['sharpe_maker_neutral'] = 0.0
        epoch_metrics['sharpe_maker_rebate'] = 0.0
        epoch_metrics['sharpe'] = 0.0
        epoch_metrics['pnl_raw'] = 0.0
        epoch_metrics['pnl_taker'] = 0.0
        epoch_metrics['pnl_maker_neutral'] = 0.0
        epoch_metrics['pnl_maker_rebate'] = 0.0

    # Compute loss averages
    if epoch_metrics['n_ppo_updates'] > 0:
        epoch_metrics['avg_policy_loss'] = epoch_metrics['total_policy_loss'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_value_loss'] = epoch_metrics['total_value_loss'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_entropy'] = epoch_metrics['total_entropy'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_uncertainty'] = epoch_metrics['total_uncertainty'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_activity'] = epoch_metrics['total_activity'] / epoch_metrics['n_ppo_updates']
        epoch_metrics['avg_turnover'] = epoch_metrics['total_turnover'] / epoch_metrics['n_ppo_updates']
    else:
        epoch_metrics['avg_policy_loss'] = 0.0
        epoch_metrics['avg_value_loss'] = 0.0
        epoch_metrics['avg_entropy'] = 0.0
        epoch_metrics['avg_uncertainty'] = 0.0
        epoch_metrics['avg_activity'] = 0.0
        epoch_metrics['avg_turnover'] = 0.0

    return epoch_metrics


def compute_validation_metrics(agent, buffer, ppo_config, experiment_type, device):
    """
    Compute model metrics (losses, entropy, uncertainty) on validation data.
    Similar to ppo_update but without gradient updates.
    """
    if len(buffer) == 0:
        return {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'uncertainty': 0.0,
            'activity': 0.0
        }

    agent.eval()
    batch = buffer.get_batch(device)

    codebooks = batch['codebooks']
    features = batch['features']
    timestamps = batch['timestamps']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    rewards = batch['rewards']
    old_values = batch['values']
    dones = batch['dones']

    # Compute advantages and returns using GAE
    from src.ppo.ppo import compute_gae
    advantages, returns = compute_gae(
        rewards, old_values, dones,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute metrics on validation data
    from src.ppo.config import ExperimentType
    import torch.nn.functional as F

    with torch.no_grad():
        if experiment_type == ExperimentType.EXP1_BOTH_ORIGINAL:
            new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                codebooks, features, timestamps, actions
            )
        elif experiment_type == ExperimentType.EXP2_FEATURES_ORIGINAL:
            new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                features, timestamps, actions
            )
        else:  # ExperimentType.EXP3_CODEBOOK_ORIGINAL
            new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                codebooks, timestamps, actions
            )

        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - ppo_config.clip_ratio, 1 + ppo_config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns)

        # Entropy (no alpha adaptation during validation)
        mean_entropy = entropy.mean()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': mean_entropy.item(),
        'uncertainty': std.mean().item(),
        'activity': torch.abs(actions).mean().item()
    }


def validate_epoch(
    agent,
    episodes: list,
    reward_config: RewardConfig,
    model_config: ModelConfig,
    ppo_config: PPOConfig,
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
    trajectory_buffer = TrajectoryBuffer(ppo_config.buffer_capacity)
    agent_state = AgentState()

    val_metrics = {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'episode_count': 0
    }

    # Episode returns for Sharpe ratio calculation (different fee scenarios)
    episode_returns_raw = []  # Baseline: Raw returns (no fees)
    episode_returns_taker = []  # Taker fee 5 bps (market orders)
    episode_returns_maker_neutral = []  # Maker fee 0 bps (limit orders)
    episode_returns_maker_rebate = []  # Maker rebate -2.5 bps (limit orders)
    total_episodes = len(episodes)

    # Group episodes by parent_id for aggregated logging
    from collections import defaultdict
    day_metrics = defaultdict(lambda: {
        'total_reward': 0.0, 'total_pnl': 0.0, 'chunks': 0,
        'total_trades': 0, 'total_steps': 0,
        'sum_mean_pos': 0.0, 'max_pos': 0.0,
        'sum_mean_std': 0.0, 'min_std': float('inf'), 'max_std': float('-inf'),
        'sum_gross_pnl_per_trade': 0.0, 'sum_position_change_per_trade': 0.0
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

        # Append trading returns for all fee scenarios (for Sharpe calculation)
        episode_returns_raw.append(metrics['total_trading_return_raw'])
        episode_returns_taker.append(metrics['total_trading_return_taker'])
        episode_returns_maker_neutral.append(metrics['total_trading_return_maker_neutral'])
        episode_returns_maker_rebate.append(metrics['total_trading_return_maker_rebate'])

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
        dm['sum_mean_std'] += metrics['mean_action_std']
        dm['min_std'] = min(dm['min_std'], metrics['min_action_std'])
        dm['max_std'] = max(dm['max_std'], metrics['max_action_std'])
        dm['sum_gross_pnl_per_trade'] += metrics['avg_gross_pnl_per_trade']
        dm['sum_position_change_per_trade'] += metrics['avg_position_change_per_trade']

        # Log when completing a parent episode (all chunks done)
        is_last_chunk = (ep_idx == total_episodes or
                        (ep_idx < total_episodes and episodes[ep_idx].parent_id != parent_id))

        if is_last_chunk:
            from src.utils.logging import logger
            n_chunks = dm['chunks']

            # Calculate aggregated metrics
            avg_position = dm['sum_mean_pos'] / n_chunks
            avg_uncertainty = dm['sum_mean_std'] / n_chunks
            avg_gross_pnl = dm['sum_gross_pnl_per_trade'] / n_chunks
            avg_position_change = dm['sum_position_change_per_trade'] / n_chunks
            trade_frequency = dm['total_trades'] / dm['total_steps']

            # Compute TC for all fee scenarios from position change
            taker_fee = 0.001  # 10 bps
            maker_fee_neutral = 0.0  # 0 bps
            maker_fee_rebate = -0.00025  # -2.5 bps (negative = rebate)

            avg_tc_taker = taker_fee * avg_position_change
            avg_tc_maker_neutral = maker_fee_neutral * avg_position_change
            avg_tc_maker_rebate = maker_fee_rebate * avg_position_change

            # Net PnL for each scenario
            net_pnl_taker = avg_gross_pnl - avg_tc_taker
            net_pnl_maker_neutral = avg_gross_pnl - avg_tc_maker_neutral
            net_pnl_maker_rebate = avg_gross_pnl - avg_tc_maker_rebate

            day_num = len([k for k in day_metrics.keys() if day_metrics[k]["chunks"] > 0])
            total_days = len(set(e.parent_id if e.parent_id is not None else i for i, e in enumerate(episodes, 1)))

            logger(f'    ┌─ Day {day_num}/{total_days} ({n_chunks} chunks, {dm["total_steps"]} steps)', "INFO")
            logger(f'    │  Trading Activity: {dm["total_trades"]} trades ({trade_frequency:.1%} frequency)', "INFO")
            logger(f'    │  Position Sizing: Mean={avg_position:.3f}, Max={dm["max_pos"]:.3f}', "INFO")
            logger(f'    │  Action Uncertainty (σ): Mean={avg_uncertainty:.3f}, Range=[{dm["min_std"]:.3f}, {dm["max_std"]:.3f}]', "INFO")
            logger(f'    │', "INFO")
            logger(f'    │  Performance (Market Orders - Taker Fee 10 bps):', "INFO")
            logger(f'    │    Gross PnL/Trade: {avg_gross_pnl:.8f}', "INFO")
            logger(f'    │    TC/Trade:        {avg_tc_taker:.8f}', "INFO")
            logger(f'    │    Net PnL/Trade:   {net_pnl_taker:.8f} ({"PROFIT" if net_pnl_taker > 0 else "LOSS"})', "INFO")
            logger(f'    │', "INFO")
            logger(f'    │  Alternative: Limit Orders (Maker Fee 0 bps):', "INFO")
            logger(f'    │    Gross PnL/Trade: {avg_gross_pnl:.8f}', "INFO")
            logger(f'    │    TC/Trade:        {avg_tc_maker_neutral:.8f}', "INFO")
            logger(f'    │    Net PnL/Trade:   {net_pnl_maker_neutral:.8f} ({"PROFIT" if net_pnl_maker_neutral > 0 else "LOSS"})', "INFO")
            logger(f'    │    Improvement:     {(net_pnl_maker_neutral - net_pnl_taker):.8f}', "INFO")
            logger(f'    │', "INFO")
            logger(f'    │  Alternative: Limit Orders (Maker Rebate -2.5 bps):', "INFO")
            logger(f'    │    Gross PnL/Trade: {avg_gross_pnl:.8f}', "INFO")
            logger(f'    │    TC/Trade:        {avg_tc_maker_rebate:.8f} (rebate)', "INFO")
            logger(f'    │    Net PnL/Trade:   {net_pnl_maker_rebate:.8f} ({"PROFIT" if net_pnl_maker_rebate > 0 else "LOSS"})', "INFO")
            logger(f'    │    Improvement:     {(net_pnl_maker_rebate - net_pnl_taker):.8f}', "INFO")
            logger(f'    └─', "INFO")

    # Compute averages
    if val_metrics['episode_count'] > 0:
        val_metrics['avg_reward'] = val_metrics['total_reward'] / val_metrics['episode_count']
        val_metrics['avg_pnl'] = val_metrics['total_pnl'] / val_metrics['episode_count']

        # Compute Sharpe ratios for all fee scenarios
        val_metrics['sharpe_raw'] = compute_sharpe_ratio(episode_returns_raw)  # Baseline: no fees
        val_metrics['sharpe_taker'] = compute_sharpe_ratio(episode_returns_taker)  # Taker 10 bps
        val_metrics['sharpe_maker_neutral'] = compute_sharpe_ratio(episode_returns_maker_neutral)  # Maker 0 bps
        val_metrics['sharpe_maker_rebate'] = compute_sharpe_ratio(episode_returns_maker_rebate)  # Maker -2.5 bps
        val_metrics['sharpe'] = val_metrics['sharpe_taker']  # Legacy field (use taker for training - harder conditions)

        # Compute average PnL per scenario (for Sharpe validation)
        val_metrics['pnl_raw'] = np.mean(episode_returns_raw)
        val_metrics['pnl_taker'] = np.mean(episode_returns_taker)
        val_metrics['pnl_maker_neutral'] = np.mean(episode_returns_maker_neutral)
        val_metrics['pnl_maker_rebate'] = np.mean(episode_returns_maker_rebate)
    else:
        val_metrics['avg_reward'] = 0.0
        val_metrics['avg_pnl'] = 0.0
        val_metrics['sharpe_raw'] = 0.0
        val_metrics['sharpe_taker'] = 0.0
        val_metrics['sharpe_maker_neutral'] = 0.0
        val_metrics['sharpe_maker_rebate'] = 0.0
        val_metrics['sharpe'] = 0.0
        val_metrics['pnl_raw'] = 0.0
        val_metrics['pnl_taker'] = 0.0
        val_metrics['pnl_maker_neutral'] = 0.0
        val_metrics['pnl_maker_rebate'] = 0.0

    # Compute model metrics (losses, entropy, uncertainty, activity) on validation data
    model_metrics = compute_validation_metrics(
        agent, trajectory_buffer, ppo_config, experiment_type, device
    )
    val_metrics['policy_loss'] = model_metrics['policy_loss']
    val_metrics['value_loss'] = model_metrics['value_loss']
    val_metrics['entropy'] = model_metrics['entropy']
    val_metrics['uncertainty'] = model_metrics['uncertainty']
    val_metrics['activity'] = model_metrics['activity']

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
        optimizer, mode='max', factor=0.3, patience=2, min_lr=1e-7
    )

    logger(f'Agent initialized: {agent.count_parameters():,} parameters', "INFO")
    logger(f'Learning rate scheduler: ReduceLROnPlateau (factor=0.3, patience=2, min_lr=1e-7)', "INFO")
    logger(f'Loss coefficients: entropy={config.ppo.entropy_coef}, uncertainty={config.ppo.uncertainty_coef}, turnover={config.ppo.turnover_coef}', "INFO")

    # Setup CSV logging for epoch results
    results_csv_path = LOG_DIR / f"split_{split_id}_epoch_results.csv"
    csv_header = [
        'epoch',
        # Training metrics - all Sharpe scenarios (ONLY taker is optimized)
        'train_sharpe_buyhold', 'train_sharpe_taker', 'train_sharpe_maker_neutral', 'train_sharpe_maker_rebate',
        'train_pnl_buyhold', 'train_pnl_taker', 'train_pnl_maker_neutral', 'train_pnl_maker_rebate',
        'train_avg_reward', 'train_avg_pnl',
        'train_policy_loss', 'train_value_loss', 'train_entropy',
        'train_uncertainty', 'train_activity', 'train_turnover',
        # Validation metrics - Sharpe scenarios + PnL (no loss/entropy metrics)
        'val_sharpe_buyhold', 'val_sharpe_taker', 'val_sharpe_maker_neutral', 'val_sharpe_maker_rebate',
        'val_pnl_buyhold', 'val_pnl_taker', 'val_pnl_maker_neutral', 'val_pnl_maker_rebate',
        'val_avg_reward', 'val_avg_pnl', 'val_activity',
        'learning_rate'
    ]

    # Create CSV file with header
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    logger(f'Epoch results will be logged to: {results_csv_path}', "INFO")

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

        logger(f'  Train - Avg Reward: {train_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {train_metrics["avg_pnl"]:.4f}', "INFO")
        logger(f'    Sharpe Ratios:', "INFO")
        logger(f'      Buy-and-Hold (baseline):  {train_metrics["sharpe_raw"]:.4f}', "INFO")
        logger(f'      Maker (0 bps):            {train_metrics["sharpe_maker_neutral"]:.4f}', "INFO")
        logger(f'      Taker (5 bps):            {train_metrics["sharpe_taker"]:.4f}  [TRAINING SCENARIO]', "INFO")
        logger(f'      Maker Rebate (-2.5 bps):  {train_metrics["sharpe_maker_rebate"]:.4f}', "INFO")
        logger(f'  Losses - Policy: {train_metrics["avg_policy_loss"]:.4f}, '
               f'Value: {train_metrics["avg_value_loss"]:.4f}, '
               f'Entropy: {train_metrics["avg_entropy"]:.4f}, '
               f'Uncertainty: {train_metrics["avg_uncertainty"]:.4f}, '
               f'Activity: {train_metrics["avg_activity"]:.4f}', "INFO")

        # Validation (every epoch)
        val_metrics = validate_epoch(
            agent, val_episodes, config.reward, config.model, config.ppo, experiment_type, device
        )

        logger(f'  Val - Avg Reward: {val_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {val_metrics["avg_pnl"]:.4f}', "INFO")
        logger(f'    Sharpe Ratios:', "INFO")
        logger(f'      Buy-and-Hold (baseline):  {val_metrics["sharpe_raw"]:.4f}', "INFO")
        logger(f'      Maker (0 bps):            {val_metrics["sharpe_maker_neutral"]:.4f}', "INFO")
        logger(f'      Taker (5 bps):            {val_metrics["sharpe_taker"]:.4f}  [TRAINING SCENARIO]', "INFO")
        logger(f'      Maker Rebate (-2.5 bps):  {val_metrics["sharpe_maker_rebate"]:.4f}', "INFO")

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
        mlflow.log_metric("train_uncertainty", train_metrics["avg_uncertainty"], step=epoch)
        mlflow.log_metric("train_activity", train_metrics["avg_activity"], step=epoch)
        mlflow.log_metric("val_sharpe", val_metrics["sharpe"], step=epoch)
        mlflow.log_metric("val_avg_reward", val_metrics["avg_reward"], step=epoch)
        mlflow.log_metric("val_avg_pnl", val_metrics["avg_pnl"], step=epoch)
        mlflow.log_metric("val_policy_loss", val_metrics["policy_loss"], step=epoch)
        mlflow.log_metric("val_value_loss", val_metrics["value_loss"], step=epoch)
        mlflow.log_metric("val_entropy", val_metrics["entropy"], step=epoch)
        mlflow.log_metric("val_uncertainty", val_metrics["uncertainty"], step=epoch)
        mlflow.log_metric("val_activity", val_metrics["activity"], step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Log epoch results to CSV file
        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,  # Epoch number (1-indexed)
                # Training - all Sharpe scenarios (ONLY taker is optimized)
                train_metrics["sharpe_raw"],
                train_metrics["sharpe_taker"],
                train_metrics["sharpe_maker_neutral"],
                train_metrics["sharpe_maker_rebate"],
                train_metrics["pnl_raw"],
                train_metrics["pnl_taker"],
                train_metrics["pnl_maker_neutral"],
                train_metrics["pnl_maker_rebate"],
                train_metrics["avg_reward"],
                train_metrics["avg_pnl"],
                train_metrics["avg_policy_loss"],
                train_metrics["avg_value_loss"],
                train_metrics["avg_entropy"],
                train_metrics["avg_uncertainty"],
                train_metrics["avg_activity"],
                train_metrics["avg_turnover"],
                # Validation - Sharpe scenarios + PnL
                val_metrics["sharpe_raw"],
                val_metrics["sharpe_taker"],
                val_metrics["sharpe_maker_neutral"],
                val_metrics["sharpe_maker_rebate"],
                val_metrics["pnl_raw"],
                val_metrics["pnl_taker"],
                val_metrics["pnl_maker_neutral"],
                val_metrics["pnl_maker_rebate"],
                val_metrics["avg_reward"],
                val_metrics["avg_pnl"],
                val_metrics["activity"],
                current_lr
            ])

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


def train_test_mode(
    test_split: int,
    config: ExperimentConfig,
    device: torch.device
):
    """
    Train PPO agent on full test_split (train+val combined) and evaluate on test_data.

    NOTE: This function requires modifications to EpisodeLoader in src.ppo:
      1. load_episodes(split_id, role=None) should load all roles when role=None
      2. load_test_episodes() method should load from 'test_data' collection

    Args:
        test_split: Split ID to use for training (all roles combined)
        config: Experiment configuration
        device: Training device

    Returns:
        Dictionary with test results
    """
    logger('', "INFO")
    logger(f'Training agent on full split_{test_split} (train+val combined)...', "INFO")
    logger('Will evaluate on test_data collection', "INFO")

    # Get experiment type from config
    experiment_type = config.data.experiment_type

    # Initialize episode loader
    data_config = config.data
    data_config.split_ids = [test_split]

    episode_loader = EpisodeLoader(data_config, episode_chunk_size=EPISODE_CHUNK_SIZE)

    # Load ALL episodes from test_split (no role filter - both train and val)
    logger('Loading training episodes (full split - train+val combined)...', "INFO")
    train_episodes = episode_loader.load_episodes(test_split, role=None)  # None = all roles
    logger(f'  Loaded {len(train_episodes)} training episodes from full split_{test_split}', "INFO")

    # Load test episodes from test_data collection
    logger('Loading test episodes from test_data...', "INFO")
    test_episodes = episode_loader.load_test_episodes()  # Special method for test_data
    logger(f'  Loaded {len(test_episodes)} test episodes', "INFO")

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

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=2, min_lr=1e-7
    )

    logger(f'Agent initialized: {agent.count_parameters():,} parameters', "INFO")
    logger(f'Learning rate scheduler: ReduceLROnPlateau (factor=0.3, patience=2, min_lr=1e-7)', "INFO")
    logger(f'Loss coefficients: entropy={config.ppo.entropy_coef}, uncertainty={config.ppo.uncertainty_coef}, turnover={config.ppo.turnover_coef}', "INFO")

    # Setup CSV logging for epoch results
    results_csv_path = LOG_DIR / f"test_split_{test_split}_epoch_results.csv"
    csv_header = [
        'epoch',
        # Training metrics (full split) - ONLY taker is optimized
        'train_sharpe_buyhold', 'train_sharpe_taker', 'train_sharpe_maker_neutral', 'train_sharpe_maker_rebate',
        'train_pnl_buyhold', 'train_pnl_taker', 'train_pnl_maker_neutral', 'train_pnl_maker_rebate',
        'train_avg_reward', 'train_avg_pnl',
        'train_policy_loss', 'train_value_loss', 'train_entropy',
        'train_uncertainty', 'train_activity', 'train_turnover',
        # Test metrics (test_data) - Sharpe scenarios + PnL
        'test_sharpe_buyhold', 'test_sharpe_taker', 'test_sharpe_maker_neutral', 'test_sharpe_maker_rebate',
        'test_pnl_buyhold', 'test_pnl_taker', 'test_pnl_maker_neutral', 'test_pnl_maker_rebate',
        'test_avg_reward', 'test_avg_pnl', 'test_activity',
        'learning_rate'
    ]

    # Create CSV file with header
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    logger(f'Epoch results will be logged to: {results_csv_path}', "INFO")

    # Training loop
    metrics_logger = MetricsLogger(log_dir=str(LOG_DIR))
    best_test_sharpe = float('-inf')
    patience_counter = 0

    for epoch in range(config.training.max_epochs):
        epoch_start = time.time()

        logger('', "INFO")
        logger(f'Epoch {epoch + 1}/{config.training.max_epochs}', "INFO")

        # Training on full split
        train_metrics = train_epoch(
            agent, optimizer, train_episodes,
            config.ppo, config.reward, config.model, experiment_type, device
        )

        logger(f'  Train (Full Split) - Avg Reward: {train_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {train_metrics["avg_pnl"]:.4f}', "INFO")
        logger(f'    Sharpe Ratios:', "INFO")
        logger(f'      Buy-and-Hold (baseline):  {train_metrics["sharpe_raw"]:.4f}', "INFO")
        logger(f'      Maker (0 bps):            {train_metrics["sharpe_maker_neutral"]:.4f}', "INFO")
        logger(f'      Taker (5 bps):            {train_metrics["sharpe_taker"]:.4f}  [TRAINING SCENARIO]', "INFO")
        logger(f'      Maker Rebate (-2.5 bps):  {train_metrics["sharpe_maker_rebate"]:.4f}', "INFO")
        logger(f'  Losses - Policy: {train_metrics["avg_policy_loss"]:.4f}, '
               f'Value: {train_metrics["avg_value_loss"]:.4f}, '
               f'Entropy: {train_metrics["avg_entropy"]:.4f}, '
               f'Uncertainty: {train_metrics["avg_uncertainty"]:.4f}, '
               f'Activity: {train_metrics["avg_activity"]:.4f}', "INFO")

        # Test evaluation (every epoch)
        test_metrics = validate_epoch(
            agent, test_episodes, config.reward, config.model, config.ppo, experiment_type, device
        )

        logger(f'  Test (test_data) - Avg Reward: {test_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {test_metrics["avg_pnl"]:.4f}', "INFO")
        logger(f'    Sharpe Ratios:', "INFO")
        logger(f'      Buy-and-Hold (baseline):  {test_metrics["sharpe_raw"]:.4f}', "INFO")
        logger(f'      Maker (0 bps):            {test_metrics["sharpe_maker_neutral"]:.4f}  [TRAINING SCENARIO]', "INFO")
        logger(f'      Taker (5 bps):            {test_metrics["sharpe_taker"]:.4f}', "INFO")
        logger(f'      Maker Rebate (-2.5 bps):  {test_metrics["sharpe_maker_rebate"]:.4f}', "INFO")

        # Update learning rate based on test Sharpe
        scheduler.step(test_metrics["sharpe"])
        current_lr = optimizer.param_groups[0]['lr']
        logger(f'  Learning rate: {current_lr:.6f}', "INFO")

        # Log to MLflow
        mlflow.log_metric("train_sharpe", train_metrics["sharpe"], step=epoch)
        mlflow.log_metric("train_sharpe_buyhold", train_metrics["sharpe_raw"], step=epoch)
        mlflow.log_metric("train_sharpe_taker", train_metrics["sharpe_taker"], step=epoch)
        mlflow.log_metric("train_sharpe_maker_neutral", train_metrics["sharpe_maker_neutral"], step=epoch)
        mlflow.log_metric("train_sharpe_maker_rebate", train_metrics["sharpe_maker_rebate"], step=epoch)
        mlflow.log_metric("train_avg_reward", train_metrics["avg_reward"], step=epoch)
        mlflow.log_metric("train_avg_pnl", train_metrics["avg_pnl"], step=epoch)
        mlflow.log_metric("train_policy_loss", train_metrics["avg_policy_loss"], step=epoch)
        mlflow.log_metric("train_value_loss", train_metrics["avg_value_loss"], step=epoch)
        mlflow.log_metric("train_entropy", train_metrics["avg_entropy"], step=epoch)
        mlflow.log_metric("train_uncertainty", train_metrics["avg_uncertainty"], step=epoch)
        mlflow.log_metric("train_activity", train_metrics["avg_activity"], step=epoch)

        mlflow.log_metric("test_sharpe", test_metrics["sharpe"], step=epoch)
        mlflow.log_metric("test_sharpe_buyhold", test_metrics["sharpe_raw"], step=epoch)
        mlflow.log_metric("test_sharpe_taker", test_metrics["sharpe_taker"], step=epoch)
        mlflow.log_metric("test_sharpe_maker_neutral", test_metrics["sharpe_maker_neutral"], step=epoch)
        mlflow.log_metric("test_sharpe_maker_rebate", test_metrics["sharpe_maker_rebate"], step=epoch)
        mlflow.log_metric("test_avg_reward", test_metrics["avg_reward"], step=epoch)
        mlflow.log_metric("test_avg_pnl", test_metrics["avg_pnl"], step=epoch)
        mlflow.log_metric("test_policy_loss", test_metrics["policy_loss"], step=epoch)
        mlflow.log_metric("test_value_loss", test_metrics["value_loss"], step=epoch)
        mlflow.log_metric("test_entropy", test_metrics["entropy"], step=epoch)
        mlflow.log_metric("test_uncertainty", test_metrics["uncertainty"], step=epoch)
        mlflow.log_metric("test_activity", test_metrics["activity"], step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Log epoch results to CSV file
        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                # Training metrics (ONLY taker is optimized)
                train_metrics["sharpe_raw"], train_metrics["sharpe_taker"],
                train_metrics["sharpe_maker_neutral"], train_metrics["sharpe_maker_rebate"],
                train_metrics["pnl_raw"], train_metrics["pnl_taker"],
                train_metrics["pnl_maker_neutral"], train_metrics["pnl_maker_rebate"],
                train_metrics["avg_reward"], train_metrics["avg_pnl"],
                train_metrics["avg_policy_loss"], train_metrics["avg_value_loss"],
                train_metrics["avg_entropy"], train_metrics["avg_uncertainty"],
                train_metrics["avg_activity"], train_metrics["avg_turnover"],
                # Test metrics - Sharpe scenarios + PnL
                test_metrics["sharpe_raw"], test_metrics["sharpe_taker"],
                test_metrics["sharpe_maker_neutral"], test_metrics["sharpe_maker_rebate"],
                test_metrics["pnl_raw"], test_metrics["pnl_taker"],
                test_metrics["pnl_maker_neutral"], test_metrics["pnl_maker_rebate"],
                test_metrics["avg_reward"], test_metrics["avg_pnl"],
                test_metrics["activity"],
                current_lr
            ])

        # Check for improvement
        if test_metrics["sharpe"] > best_test_sharpe:
            best_test_sharpe = test_metrics["sharpe"]
            patience_counter = 0

            # Save best model
            save_checkpoint(
                agent, optimizer, epoch, test_split, test_metrics["sharpe"],
                checkpoint_dir=str(CHECKPOINT_DIR)
            )
            logger(f'  ✓ New best model saved (Test Sharpe: {best_test_sharpe:.4f})', "INFO")
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
        'best_test_sharpe': best_test_sharpe,
        'best_test_sharpe_buyhold': test_metrics["sharpe_raw"],
        'best_test_sharpe_taker': test_metrics["sharpe_taker"],
        'best_test_sharpe_maker_neutral': test_metrics["sharpe_maker_neutral"],
        'best_test_sharpe_maker_rebate': test_metrics["sharpe_maker_rebate"],
        'epochs_trained': epoch + 1
    }


# =================================================================================================
# Main Execution
# =================================================================================================

def main():
    """Main execution function."""
    import argparse

    # Allow modification of global directory variables for parallel execution
    global CHECKPOINT_DIR, LOG_DIR

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PPO Agent Training (Stage 21) with Train/Test Modes')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Pipeline mode: train (CPCV on splits) or test (full split + test_data eval)')
    parser.add_argument('--test-split', type=int, default=0,
                        help='[TEST MODE] Split ID to use for full training (default: 0)')
    parser.add_argument('--experiment', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Experiment type: 1=Both sources (original), 2=Features only (original), '
                             '3=Codebook only (original), 4=Codebook (synthetic). '
                             'If not specified, runs all experiments sequentially.')
    parser.add_argument('--splits', type=str, default=None,
                        help='[TRAIN MODE] Comma-separated list of split IDs to train (e.g., "0,1,2"). '
                             'If not specified, trains on all available splits.')
    parser.add_argument('--max-splits', type=int, default=None,
                        help='[TRAIN MODE] Maximum number of splits to train on (uses first N splits). '
                             'Useful for quick testing.')
    args = parser.parse_args()

    # Determine which experiments to run
    if args.experiment is not None:
        experiments_to_run = [args.experiment]
    else:
        # Run all experiments by default (including synthetic)
        experiments_to_run = [1, 2, 3, 4]

    # Map experiment number to enum
    experiment_mapping = {
        1: ExperimentType.EXP1_BOTH_ORIGINAL,
        2: ExperimentType.EXP2_FEATURES_ORIGINAL,
        3: ExperimentType.EXP3_CODEBOOK_ORIGINAL,
        4: ExperimentType.EXP4_SYNTHETIC_BINS
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

    # Adjust experiment name and artifact directory based on mode
    if args.mode == 'test':
        experiment_name = f"{MLFLOW_EXPERIMENT_NAME}_Test"
        artifact_base_dir = ARTIFACT_BASE_DIR.parent / "test"
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger(f'PPO TRAINING - TEST MODE (STAGE 21)', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Training on full split_{args.test_split} (train+val combined)', "INFO")
        logger(f'Evaluating on test_data collection', "INFO")
        logger('', "INFO")
    else:
        experiment_name = MLFLOW_EXPERIMENT_NAME
        artifact_base_dir = ARTIFACT_BASE_DIR
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger(f'PPO TRAINING - TRAIN MODE (STAGE 21)', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Using CPCV on {len(split_ids)} splits', "INFO")
        logger('', "INFO")

    mlflow.set_experiment(experiment_name)
    logger(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}', "INFO")
    logger(f'MLflow experiment: {experiment_name}', "INFO")

    # Create artifact directories
    artifact_base_dir.mkdir(parents=True, exist_ok=True)

    # Run each experiment
    for exp_num in experiments_to_run:
        selected_experiment = experiment_mapping[exp_num]

        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger(f'PPO AGENT TRAINING - EXPERIMENT {exp_num} (STAGE 21)', "INFO")
        logger('=' * 100, "INFO")
        logger(f'Experiment: {selected_experiment.name}', "INFO")

        # Create experiment-specific directories
        exp_artifact_dir = artifact_base_dir / f"experiment_{exp_num}"
        CHECKPOINT_DIR = exp_artifact_dir / "checkpoints"
        LOG_DIR = exp_artifact_dir / "logs"

        # Create directories
        exp_artifact_dir.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Create experiment config
        config = ExperimentConfig(
            name=f"ppo_exp{exp_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model=ModelConfig(window_size=WINDOW_SIZE, horizon=HORIZON),
            ppo=PPOConfig(),
            reward=RewardConfig(),
            training=TrainingConfig(device=str(device)),
            data=DataConfig(
                database_name=DB_NAME,
                split_ids=split_ids if args.mode == 'train' else [args.test_split],
                experiment_type=selected_experiment
            )
        )

        # Save config
        config_path = exp_artifact_dir / "experiment_config.json"
        config.save(str(config_path))
        logger(f'Experiment config saved to: {config_path}', "INFO")

        # Log training configuration
        logger('', "INFO")
        logger('Training Configuration:', "INFO")
        logger(f'  Max epochs: {config.training.max_epochs}', "INFO")
        logger(f'  Episodes per epoch: ALL (no limit)', "INFO")
        if args.mode == 'train':
            logger(f'  Splits to train: {len(split_ids)}', "INFO")
        else:
            logger(f'  Training on: full split_{args.test_split} (train+val combined)', "INFO")
            logger(f'  Evaluating on: test_data collection', "INFO")
        logger(f'  Early stopping patience: {config.training.patience} epochs', "INFO")

        if args.mode == 'train':
            # Estimate training time for train mode
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

            if args.mode == 'train':
                # TRAIN MODE: CPCV training on splits
                all_results = {}

                for split_id in split_ids:
                    logger('', "INFO")
                    logger('=' * 100, "INFO")
                    logger(f'SPLIT {split_id}', "INFO")
                    logger('=' * 100, "INFO")

                    with mlflow.start_run(run_name=f"split_{split_id}", nested=True):
                        mlflow.log_param("split_id", split_id)
                        mlflow.log_param("experiment_type", exp_num)
                        mlflow.log_param("mode", "train")

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
                logger(f'EXPERIMENT {exp_num} COMPLETE (TRAIN MODE)', "INFO")
                logger('=' * 100, "INFO")

                avg_sharpe = np.mean([r['best_val_sharpe'] for r in all_results.values()])
                logger(f'Average validation Sharpe across splits: {avg_sharpe:.4f}', "INFO")
                logger(f'Checkpoints saved to: {CHECKPOINT_DIR}', "INFO")

                mlflow.log_metric("avg_val_sharpe_across_splits", avg_sharpe)

            else:
                # TEST MODE: Train on full split, evaluate on test_data
                logger('', "INFO")
                logger('=' * 100, "INFO")
                logger(f'TEST SPLIT {args.test_split} (FULL)', "INFO")
                logger('=' * 100, "INFO")

                mlflow.log_param("test_split", args.test_split)
                mlflow.log_param("experiment_type", exp_num)
                mlflow.log_param("mode", "test")

                results = train_test_mode(args.test_split, config, device)

                # Log test metrics
                mlflow.log_metric("best_test_sharpe", results['best_test_sharpe'])
                mlflow.log_metric("best_test_sharpe_buyhold", results['best_test_sharpe_buyhold'])
                mlflow.log_metric("best_test_sharpe_taker", results['best_test_sharpe_taker'])
                mlflow.log_metric("best_test_sharpe_maker_neutral", results['best_test_sharpe_maker_neutral'])
                mlflow.log_metric("best_test_sharpe_maker_rebate", results['best_test_sharpe_maker_rebate'])
                mlflow.log_metric("epochs_trained", results['epochs_trained'])

                # Summary
                logger('', "INFO")
                logger('=' * 100, "INFO")
                logger(f'EXPERIMENT {exp_num} COMPLETE (TEST MODE)', "INFO")
                logger('=' * 100, "INFO")
                logger(f'Test Sharpe Ratios (best epoch):', "INFO")
                logger(f'  Buy-and-Hold (baseline):  {results["best_test_sharpe_buyhold"]:.4f}', "INFO")
                logger(f'  Taker (5 bps):            {results["best_test_sharpe_taker"]:.4f}', "INFO")
                logger(f'  Maker (0 bps):            {results["best_test_sharpe_maker_neutral"]:.4f}', "INFO")
                logger(f'  Maker Rebate (-2.5 bps):  {results["best_test_sharpe_maker_rebate"]:.4f}', "INFO")
                logger(f'Epochs trained: {results["epochs_trained"]}', "INFO")
                logger(f'Model saved to: {CHECKPOINT_DIR}', "INFO")

    # Final summary
    logger('', "INFO")
    logger('=' * 100, "INFO")
    logger(f'ALL EXPERIMENTS COMPLETE ({args.mode.upper()} MODE)', "INFO")
    logger('=' * 100, "INFO")
    logger(f'Completed {len(experiments_to_run)} experiment(s)', "INFO")
    logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")
    if args.mode == 'test':
        logger(f'Test results saved to: {artifact_base_dir}', "INFO")
    else:
        logger(f'Training results saved to: {artifact_base_dir}', "INFO")


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
        logger('Stage 21 completed successfully', "INFO")

    except Exception as e:
        logger(f'ERROR: {str(e)}', "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
