"""PPO training algorithm."""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .buffer import TrajectoryBuffer


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (T,) tensor of rewards
        values: (T,) tensor of value estimates
        dones: (T,) tensor of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: (T,) tensor of advantages
        returns: (T,) tensor of returns (targets for value function)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # Compute advantages in reverse (backward through time)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        # TD error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    
    # Returns are advantages + values
    returns = advantages + values
    
    return advantages, returns


def ppo_update(
    agent,
    buffer: TrajectoryBuffer,
    optimizer: torch.optim.Optimizer,
    config,
    experiment_type,
    device: str = 'cuda',
    entropy_coef_override: float = None
) -> Dict[str, float]:
    """
    Perform PPO update using trajectories in buffer.

    Args:
        agent: Actor-critic model (Transformer, Features, or Codebook)
        experiment_type: ExperimentType enum to determine input format
        buffer: TrajectoryBuffer with collected experience
        optimizer: Optimizer
        config: PPOConfig with hyperparameters
        device: Device for computation
        entropy_coef_override: Optional override for entropy coefficient (for annealing)

    Returns:
        Dictionary with loss metrics
    """
    # Get batch
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
    # returns = raw discounted rewards (NOT normalized)
    # advantages = how much better/worse actions were vs baseline
    advantages, returns = compute_gae(
        rewards, old_values, dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda
    )

    # Normalize advantages for policy gradient (stabilizes training)
    advantages_mean = advantages.mean()
    advantages_std = advantages.std()
    advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

    # Store raw return statistics for diagnostics
    returns_mean = returns.mean()
    returns_std = returns.std()

    # Normalize returns for value learning (stabilizes training)
    # Policy still uses normalized advantages (computed from raw returns via GAE)
    # This prevents value loss divergence while keeping policy incentives unchanged
    if returns_std > 1e-8:
        returns_normalized = (returns - returns_mean) / (returns_std + 1e-8)
    else:
        returns_normalized = returns - returns_mean

    # PPO epochs
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_uncertainty = 0
    total_activity = 0
    total_turnover = 0       # Track position changes (turnover)
    total_clip_fraction = 0  # Track how often clipping activates
    total_approx_kl = 0      # Track KL divergence
    n_updates = 0
    
    for epoch in range(config.n_epochs):
        # Create sequential minibatches (no shuffling to preserve temporal order)
        T = len(actions)
        
        for start in range(0, T, config.batch_size):
            end = min(start + config.batch_size, T)

            # Extract minibatch
            mb_codebooks = codebooks[start:end]
            # Handle None features for codebook-only experiments
            mb_features = features[start:end] if features is not None else None
            mb_timestamps = timestamps[start:end]
            mb_actions = actions[start:end]
            mb_old_log_probs = old_log_probs[start:end]
            mb_advantages = advantages[start:end]
            mb_returns_normalized = returns_normalized[start:end]  # Normalized for value learning

            # Forward pass - call evaluate_actions with correct arguments based on experiment
            from src.ppo.config import ExperimentType

            if experiment_type == ExperimentType.EXP1_BOTH_ORIGINAL:
                # Experiment 1: Both codebook + features
                new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                    mb_codebooks, mb_features, mb_timestamps, mb_actions
                )
            elif experiment_type == ExperimentType.EXP2_FEATURES_ORIGINAL:
                # Experiment 2: Features only
                new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                    mb_features, mb_timestamps, mb_actions
                )
            elif experiment_type == ExperimentType.EXP4_SYNTHETIC_BINS:
                # Experiment 4: Codebook only (same as Experiment 3, but synthetic data)
                new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                    mb_codebooks, mb_timestamps, mb_actions
                )
            else:  # ExperimentType.EXP3_CODEBOOK_ORIGINAL
                # Experiment 3: Codebook only
                new_log_probs, new_values, entropy, std = agent.evaluate_actions(
                    mb_codebooks, mb_timestamps, mb_actions
                )
            
            # Policy loss (PPO clipped objective)
            # Uses normalized advantages for stable gradients
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Diagnostic: Track clipping fraction (how often ratio is clipped)
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > config.clip_ratio).float()).item()

            # Diagnostic: Approximate KL divergence (for early stopping)
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

            # Value loss (MSE on normalized returns)
            # Normalization prevents divergence from unbounded raw returns
            # Policy gradient is unchanged (uses normalized advantages from raw returns via GAE)
            value_loss = F.mse_loss(new_values, mb_returns_normalized)

            # Entropy bonus (encourages exploration, can be annealed)
            current_entropy_coef = entropy_coef_override if entropy_coef_override is not None else config.entropy_coef
            entropy_bonus = current_entropy_coef * entropy.mean()

            # Uncertainty penalty (prevents std exploitation)
            uncertainty_penalty = config.uncertainty_coef * std.mean()

            # Turnover penalty (penalizes frequent position changes to encourage selective trading)
            # Compute position changes: |action[t] - action[t-1]|
            # Higher penalty → fewer trades → higher quality trades
            if mb_actions.shape[0] > 1:
                turnover = torch.abs(mb_actions[1:] - mb_actions[:-1]).mean()
            else:
                turnover = torch.tensor(0.0, device=mb_actions.device)
            turnover_penalty = config.turnover_coef * turnover

            # Total loss (policy + value - entropy + turnover + uncertainty)
            # Subtract bonuses, add penalties
            loss = (policy_loss +
                   config.value_coef * value_loss -
                   entropy_bonus +
                   turnover_penalty +
                   uncertainty_penalty)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_uncertainty += std.mean().item()
            total_activity += torch.abs(mb_actions).mean().item()  # Still track activity for interpretability
            total_turnover += turnover.item()  # Track turnover (position changes)
            total_clip_fraction += clip_fraction
            total_approx_kl += approx_kl
            n_updates += 1

    return {
        'policy_loss': total_policy_loss / n_updates,
        'value_loss': total_value_loss / n_updates,
        'entropy': total_entropy / n_updates,
        'uncertainty': total_uncertainty / n_updates,
        'activity': total_activity / n_updates,  # Average |actions|
        'turnover': total_turnover / n_updates,  # Average position changes
        'clip_fraction': total_clip_fraction / n_updates,  # How often clipping occurs
        'approx_kl': total_approx_kl / n_updates,  # Approximate KL divergence
        'advantages_mean': advantages_mean.item(),  # Raw advantage statistics
        'advantages_std': advantages_std.item(),
        'returns_mean': returns_mean.item(),  # Raw return statistics
        'returns_std': returns_std.item(),
        'n_updates': n_updates
    }