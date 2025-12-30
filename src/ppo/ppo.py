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
    log_alpha: torch.Tensor = None,
    alpha_optimizer: torch.optim.Optimizer = None
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
    # NOTE: returns are NOT normalized - value function learns raw reward scale
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO epochs
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_alpha = 0
    n_updates = 0
    
    for epoch in range(config.n_epochs):
        # Create sequential minibatches (no shuffling to preserve temporal order)
        T = len(actions)
        
        for start in range(0, T, config.batch_size):
            end = min(start + config.batch_size, T)
            
            # Extract minibatch
            mb_codebooks = codebooks[start:end]
            mb_features = features[start:end]
            mb_timestamps = timestamps[start:end]
            mb_actions = actions[start:end]
            mb_old_log_probs = old_log_probs[start:end]
            mb_advantages = advantages[start:end]
            mb_returns = returns[start:end]

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

            # Value loss (MSE)
            # Uses RAW (unnormalized) returns - value function predicts actual reward scale
            value_loss = F.mse_loss(new_values, mb_returns)

            # Adaptive entropy (SAC-style)
            # Compute current alpha (temperature parameter)
            if log_alpha is not None:
                alpha = log_alpha.exp()

                # Entropy target loss (updates alpha to maintain target entropy)
                mean_entropy = entropy.mean()
                alpha_loss = (alpha * (mean_entropy - config.target_entropy).detach()).mean()

                # Update alpha
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # Use adaptive alpha in loss (detach to prevent gradient flow to alpha)
                entropy_loss = -alpha.detach() * mean_entropy
            else:
                # Fallback to fixed entropy coefficient (for validation)
                entropy_loss = -0.05 * entropy.mean()
                alpha = torch.tensor(0.05)

            # Total loss (removed uncertainty penalty)
            loss = (policy_loss +
                   config.value_coef * value_loss +
                   entropy_loss)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_alpha += alpha.item()
            n_updates += 1

    return {
        'policy_loss': total_policy_loss / n_updates,
        'value_loss': total_value_loss / n_updates,
        'entropy': total_entropy / n_updates,
        'alpha': total_alpha / n_updates,
        'n_updates': n_updates
    }