"""Reward function computation."""

import torch
from typing import Optional


def compute_forward_looking_reward(
    action_prev: float,
    action_curr: float,
    future_returns: torch.Tensor,
    volatility: float,
    spread_bps: float = 5.0,
    tc_bps: float = 2.5,
    lambda_risk: float = 1.0,
    alpha_penalty: float = 0.01,
    epsilon: float = 1e-8
) -> float:
    """
    Compute forward-looking information ratio reward.
    
    Args:
        action_prev: Previous position a_{t-1}
        action_curr: Current position a_t
        future_returns: (H,) tensor of future log-returns
        volatility: Current market volatility (from features)
        spread_bps: Spread cost in basis points
        tc_bps: Transaction cost in basis points
        lambda_risk: Risk adjustment weight
        alpha_penalty: Position size penalty coefficient
        epsilon: Numerical stability constant
    
    Returns:
        Reward value
    """
    # Expected future return (what a_t will earn)
    expected_return = future_returns.mean().item()
    
    # Position-weighted expected PnL
    expected_pnl = action_curr * expected_return
    
    # Transaction costs (paid when changing position)
    position_change = abs(action_curr - action_prev)
    total_tc = (spread_bps + tc_bps) / 10000.0 * position_change
    
    # Active return (net of costs)
    active_return = expected_pnl - total_tc
    
    # Tracking risk (volatility scaled by position size)
    tracking_risk = volatility * (abs(action_curr) + epsilon)
    
    # Information ratio component (raw, unbounded)
    info_ratio_raw = active_return / tracking_risk

    # Apply tanh to bound the ratio to [-1, 1] and prevent explosions
    # This prevents extreme rewards when volatility is very low
    info_ratio = torch.tanh(torch.tensor(info_ratio_raw)).item()

    # Position size penalty (regularization)
    position_penalty = alpha_penalty * (action_curr ** 2)

    # Final reward - scale info_ratio by 10 to maintain learning signal
    # Typical range: [-10, +10] instead of unbounded [-500, +500]
    reward = lambda_risk * 10.0 * info_ratio - position_penalty

    return reward


def compute_transaction_cost(
    action_prev: float,
    action_curr: float,
    spread_bps: float = 5.0,
    tc_bps: float = 2.5
) -> float:
    """
    Compute transaction cost for position change.
    
    Args:
        action_prev: Previous position
        action_curr: Current position
        spread_bps: Spread cost in basis points
        tc_bps: Transaction cost in basis points
    
    Returns:
        Transaction cost (as decimal, e.g., 0.000075 for 0.75 bps)
    """
    position_change = abs(action_curr - action_prev)
    return (spread_bps + tc_bps) / 10000.0 * position_change


def compute_unrealized_pnl(
    position: float,
    future_returns: torch.Tensor
) -> float:
    """
    Compute unrealized PnL from current position.
    
    Args:
        position: Current position size
        future_returns: (H,) tensor of expected future returns
    
    Returns:
        Unrealized PnL
    """
    expected_return = future_returns.mean().item()
    return position * expected_return