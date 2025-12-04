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
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Perform PPO update using trajectories in buffer.
    
    Args:
        agent: ActorCriticTransformer model
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
    
    # Compute advantages using GAE
    advantages, returns = compute_gae(
        rewards, old_values, dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda
    )
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO epochs
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
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
            
            # Forward pass
            new_log_probs, new_values, entropy = agent.evaluate_actions(
                mb_codebooks, mb_features, mb_timestamps, mb_actions
            )
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (MSE)
            value_loss = F.mse_loss(new_values, mb_returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1
    
    return {
        'policy_loss': total_policy_loss / n_updates,
        'value_loss': total_value_loss / n_updates,
        'entropy': total_entropy / n_updates,
        'n_updates': n_updates
    }