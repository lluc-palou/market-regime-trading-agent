"""Reward function computation."""

import torch
from typing import Optional


def compute_policy_based_position(
    action: float,
    action_std: float,
    confidence_weight: float = 1.0,
    min_std: float = 0.1,
    max_std: float = 3.0
) -> float:
    """
    Compute position using agent's policy distribution (action mean and std).

    This allows the PPO algorithm to learn position sizing directly:
    - action (after tanh): direction ∈ [-1, 1]
    - action_std: agent's uncertainty/confidence
    - Lower std → higher confidence → larger position
    - Higher std → lower confidence → smaller position

    Formula: position = action × exp(-confidence_weight × (std - min_std))

    Args:
        action: Agent action (tanh-squashed, in [-1, 1])
        action_std: Policy standard deviation (agent's uncertainty)
        confidence_weight: How strongly std affects position size (default: 1.0)
        min_std: Minimum expected std for scaling (default: 0.1)
        max_std: Maximum expected std for capping (default: 3.0)

    Returns:
        Position in range approximately [-1, 1]

    Examples:
        With confidence_weight=1.0, min_std=0.1:
        - action=0.5, std=0.1 (very confident) → position ≈ 0.5 × exp(0) = 0.50
        - action=0.5, std=0.5 (uncertain) → position ≈ 0.5 × exp(-0.4) = 0.34
        - action=0.5, std=1.0 (very uncertain) → position ≈ 0.5 × exp(-0.9) = 0.20
        - action=0.5, std=2.0 (extremely uncertain) → position ≈ 0.5 × exp(-1.9) = 0.08
    """
    # Clamp std to reasonable range
    std_clamped = max(min_std, min(action_std, max_std))

    # Confidence scaling: exp(-k × (std - min_std))
    # When std = min_std → scale = 1.0 (maximum position)
    # When std increases → scale decreases exponentially
    confidence_scale = torch.exp(torch.tensor(-confidence_weight * (std_clamped - min_std))).item()

    # Compute position
    position = action * confidence_scale

    return position


def compute_volatility_scaled_position(
    action: float,
    volatility: float,
    vol_constant: float = 0.05,
    epsilon: float = 0.1
) -> float:
    """
    [DEPRECATED] Scale agent's action to actual position using volatility.

    NOTE: This function decouples position sizing from agent learning, which prevents
    the PPO algorithm from learning optimal position sizing. Use compute_policy_based_position()
    instead to allow the agent to learn both direction and sizing.

    Formula: position = action × min(1.0, C / max(σ, epsilon))

    This decouples position sizing from agent learning:
    - Agent learns direction and conviction (action)
    - Position sizing handled by volatility (risk management)
    - Volatility floor (epsilon) prevents extreme positions when σ near zero

    Args:
        action: Raw agent action (can be any value, positive or negative)
        volatility: Realized volatility from features (standardized/normalized)
        vol_constant: Position sizing constant C (default: 0.05 for ~5-6% positions)
        epsilon: Volatility floor to prevent explosions (default: 0.1)

    Returns:
        Scaled position (can be long or short)

    Examples:
        With C=0.05, epsilon=0.1:
        - σ = 0.8 (typical) → scaling = 0.05/0.8 = 0.0625 → ~6% position
        - σ = 0.001 (rare) → floor to 0.1 → scaling = 0.05/0.1 = 0.5 → 50% position
        - σ = 3.0 (high) → scaling = 0.05/3.0 = 0.0167 → ~2% position
    """
    # Floor volatility to prevent near-zero values from causing position explosions
    vol_safe = max(abs(volatility), epsilon)

    # Volatility scaling: smaller position when vol is high
    vol_scaling = min(1.0, vol_constant / vol_safe)

    # Apply scaling to action
    position = action * vol_scaling

    return position


def compute_simple_reward(
    position_prev: float,
    position_curr: float,
    target: float,
    taker_fee: float = 0.0005,
    epsilon: float = 1e-8
) -> tuple[float, float, float]:
    """
    Compute simple PnL-based reward.

    Formula: reward = position × target - taker_fee × |Δposition|

    Args:
        position_prev: Previous position (after vol scaling)
        position_curr: Current position (after vol scaling)
        target: One-step forward log return (immediate reward signal)
        taker_fee: Transaction cost as decimal (default: 0.0005 = 5 bps)
        epsilon: Numerical stability

    Returns:
        Tuple of (reward, gross_pnl, transaction_cost)
    """
    # Gross PnL from holding position
    gross_pnl = position_curr * target

    # Transaction cost (only charged when position changes)
    position_change = abs(position_curr - position_prev)
    tc = taker_fee * position_change if position_change > epsilon else 0.0

    # Net reward
    reward = gross_pnl - tc

    return reward, gross_pnl, tc


def compute_transaction_cost(
    position_prev: float,
    position_curr: float,
    taker_fee: float = 0.0005,
    epsilon: float = 1e-8
) -> float:
    """
    Compute transaction cost for position change.

    Args:
        position_prev: Previous position
        position_curr: Current position
        taker_fee: Transaction cost as decimal (default: 0.0005 = 5 bps)
        epsilon: Numerical stability

    Returns:
        Transaction cost (as decimal)
    """
    position_change = abs(position_curr - position_prev)
    return taker_fee * position_change if position_change > epsilon else 0.0


def compute_unrealized_pnl(
    position: float,
    target: float
) -> float:
    """
    Compute unrealized PnL from current position.

    Args:
        position: Current position size (after vol scaling)
        target: One-step forward log return

    Returns:
        Unrealized PnL
    """
    return position * target
