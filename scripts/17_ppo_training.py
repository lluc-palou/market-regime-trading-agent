"""
PPO Agent Training Script (Stage 17)

Trains PPO agents for limit order book trading using VQ-VAE latent representations
and hand-crafted features.

This is Stage 17 in the pipeline - follows VQ-VAE production training (Stage 14).

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
    python scripts/17_ppo_training.py
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
    EpisodeLoader,
    StateBuffer,
    TrajectoryBuffer,
    AgentState,
    Transition,
    ppo_update,
    get_valid_timesteps,
    compute_forward_looking_reward,
    compute_transaction_cost,
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
MAX_EPISODES_PER_EPOCH = 50  # Limit episodes per epoch for manageable training
VALIDATE_EVERY = 5  # Validate every N epochs

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
    agent: ActorCriticTransformer,
    episode,
    state_buffer: StateBuffer,
    trajectory_buffer: TrajectoryBuffer,
    agent_state: AgentState,
    reward_config: RewardConfig,
    model_config: ModelConfig,
    device: str,
    deterministic: bool = False
):
    """
    Run one episode and collect trajectories.

    Args:
        agent: PPO agent model
        episode: Episode object with samples
        state_buffer: Rolling window state buffer
        trajectory_buffer: Trajectory buffer for PPO updates
        agent_state: Agent trading state
        reward_config: Reward function config
        model_config: Model config
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

        # Agent selects action
        with torch.no_grad():
            action, log_prob, value = agent.act(
                codebooks, features, timestamps,
                deterministic=deterministic
            )

        action_val = action.item()
        log_prob_val = log_prob.item() if not deterministic else 0.0
        value_val = value.item()

        # Get future returns for reward computation
        future_returns = torch.tensor([
            episode.samples[t + h]['target'] for h in range(1, model_config.horizon + 1)
        ], dtype=torch.float32)

        # Get volatility from features (assume it's in features)
        volatility = current_sample['features'][1].item()  # Assuming 2nd feature is volatility

        # Compute reward
        prev_action = agent_state.current_position
        reward = compute_forward_looking_reward(
            action_prev=prev_action,
            action_curr=action_val,
            future_returns=future_returns,
            volatility=volatility,
            spread_bps=reward_config.spread_bps,
            tc_bps=reward_config.tc_bps,
            lambda_risk=reward_config.lambda_risk,
            alpha_penalty=reward_config.alpha_penalty,
            epsilon=reward_config.epsilon
        )

        # Transaction cost
        tc = compute_transaction_cost(
            prev_action, action_val,
            reward_config.spread_bps, reward_config.tc_bps
        )

        # Unrealized PnL
        unrealized = compute_unrealized_pnl(action_val, future_returns)

        # Update agent state
        log_return = current_sample['target']
        agent_state.update(action_val, log_return, tc, reward, unrealized)
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
    agent: ActorCriticTransformer,
    optimizer: torch.optim.Optimizer,
    episodes: list,
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    model_config: ModelConfig,
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
        'avg_episode_length': 0.0
    }

    episode_returns = []

    # Limit episodes per epoch for manageable training
    episodes_to_run = episodes[:MAX_EPISODES_PER_EPOCH]

    for episode in episodes_to_run:
        metrics = run_episode(
            agent, episode, state_buffer, trajectory_buffer, agent_state,
            reward_config, model_config, device, deterministic=False
        )

        if metrics is None:
            continue

        epoch_metrics['total_reward'] += metrics['total_reward']
        epoch_metrics['total_pnl'] += metrics['total_pnl']
        epoch_metrics['episode_count'] += 1
        epoch_metrics['avg_episode_length'] += metrics['episode_length']
        episode_returns.append(metrics['total_reward'])

        # Perform PPO update if buffer is full
        if trajectory_buffer.is_full():
            loss_metrics = ppo_update(
                agent, trajectory_buffer, optimizer, ppo_config, device
            )
            trajectory_buffer.clear()

    # Final PPO update with remaining trajectories
    if len(trajectory_buffer) > 0:
        loss_metrics = ppo_update(
            agent, trajectory_buffer, optimizer, ppo_config, device
        )
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

    return epoch_metrics


def validate_epoch(
    agent: ActorCriticTransformer,
    episodes: list,
    reward_config: RewardConfig,
    model_config: ModelConfig,
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

    for episode in episodes:
        metrics = run_episode(
            agent, episode, state_buffer, trajectory_buffer, agent_state,
            reward_config, model_config, device, deterministic=True
        )

        if metrics is None:
            continue

        val_metrics['total_reward'] += metrics['total_reward']
        val_metrics['total_pnl'] += metrics['total_pnl']
        val_metrics['episode_count'] += 1
        episode_returns.append(metrics['total_reward'])

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

    # Initialize episode loader
    data_config = DataConfig(
        mongodb_uri=MONGO_URI,
        database_name=DB_NAME,
        split_ids=[split_id],
        role_train='train',
        role_val='validation'
    )

    episode_loader = EpisodeLoader(data_config)

    # Load episodes
    logger('Loading training episodes...', "INFO")
    train_episodes = episode_loader.load_episodes(split_id, role='train')
    logger(f'  Loaded {len(train_episodes)} training episodes', "INFO")

    logger('Loading validation episodes...', "INFO")
    val_episodes = episode_loader.load_episodes(split_id, role='validation')
    logger(f'  Loaded {len(val_episodes)} validation episodes', "INFO")

    # Initialize agent
    agent = ActorCriticTransformer(config.model).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config.ppo.learning_rate,
        weight_decay=config.ppo.weight_decay
    )

    logger(f'Agent initialized: {agent.count_parameters():,} parameters', "INFO")

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
            config.ppo, config.reward, config.model, device
        )

        logger(f'  Train - Sharpe: {train_metrics["sharpe"]:.4f}, '
               f'Avg Reward: {train_metrics["avg_reward"]:.4f}, '
               f'Avg PnL: {train_metrics["avg_pnl"]:.4f}', "INFO")

        # Validation (every N epochs)
        if (epoch + 1) % VALIDATE_EVERY == 0:
            val_metrics = validate_epoch(
                agent, val_episodes, config.reward, config.model, device
            )

            logger(f'  Val - Sharpe: {val_metrics["sharpe"]:.4f}, '
                   f'Avg Reward: {val_metrics["avg_reward"]:.4f}, '
                   f'Avg PnL: {val_metrics["avg_pnl"]:.4f}', "INFO")

            # Log to MLflow
            mlflow.log_metric("train_sharpe", train_metrics["sharpe"], step=epoch)
            mlflow.log_metric("train_avg_reward", train_metrics["avg_reward"], step=epoch)
            mlflow.log_metric("val_sharpe", val_metrics["sharpe"], step=epoch)
            mlflow.log_metric("val_avg_reward", val_metrics["avg_reward"], step=epoch)

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
        else:
            # Log only training metrics
            mlflow.log_metric("train_sharpe", train_metrics["sharpe"], step=epoch)
            mlflow.log_metric("train_avg_reward", train_metrics["avg_reward"], step=epoch)

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
    logger('=' * 100, "INFO")
    logger('PPO AGENT TRAINING (STAGE 17)', "INFO")
    logger('=' * 100, "INFO")

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

    # Create experiment config
    config = ExperimentConfig(
        name=f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model=ModelConfig(window_size=WINDOW_SIZE, horizon=HORIZON),
        ppo=PPOConfig(),
        reward=RewardConfig(),
        training=TrainingConfig(device=str(device)),
        data=DataConfig(database_name=DB_NAME, split_ids=split_ids)
    )

    # Save config
    config_path = ARTIFACT_BASE_DIR / "experiment_config.json"
    config.save(str(config_path))
    logger(f'Experiment config saved to: {config_path}', "INFO")

    # Main training loop
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

                results = train_split(split_id, config, device)
                all_results[split_id] = results

                mlflow.log_metric("best_val_sharpe", results['best_val_sharpe'])
                mlflow.log_metric("epochs_trained", results['epochs_trained'])

                logger('', "INFO")
                logger(f'Split {split_id} complete:', "INFO")
                logger(f'  Best validation Sharpe: {results["best_val_sharpe"]:.4f}', "INFO")
                logger(f'  Epochs trained: {results["epochs_trained"]}', "INFO")

        # Summary
        logger('', "INFO")
        logger('=' * 100, "INFO")
        logger('TRAINING COMPLETE', "INFO")
        logger('=' * 100, "INFO")

        avg_sharpe = np.mean([r['best_val_sharpe'] for r in all_results.values()])
        logger(f'Average validation Sharpe across splits: {avg_sharpe:.4f}', "INFO")
        logger(f'Checkpoints saved to: {CHECKPOINT_DIR}', "INFO")
        logger(f'MLflow tracking: {MLFLOW_TRACKING_URI}', "INFO")

        mlflow.log_metric("avg_val_sharpe_across_splits", avg_sharpe)


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
