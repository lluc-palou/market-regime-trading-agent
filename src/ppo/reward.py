"""Reward function computation."""

import torch
import numpy as np
import math
from typing import Optional


def compute_ewma_volatility(
    recent_targets: list,
    half_life: int = 20,
    min_samples: int = 5,
    default_vol: float = 0.002
) -> float:
    """Compute EWMA volatility estimate from recent return samples."""
    if len(recent_targets) < min_samples:
        return default_vol

    alpha = 1 - math.exp(math.log(0.5) / half_life)
    variance = float(np.var(recent_targets))

    for ret in reversed(recent_targets):
        variance = alpha * (ret ** 2) + (1 - alpha) * variance

    return max(math.sqrt(variance), 1e-5)


def compute_policy_based_position(
    action: float,
    action_std: float,
    confidence_weight: float = 1.0,
    min_std: float = 0.1,
    max_std: float = 100.0
) -> float:
    """Compute position using agent's policy distribution (mean and std)."""
    std_clamped = max(min_std, min(action_std, max_std))
    confidence_scale = torch.exp(torch.tensor(-confidence_weight * (std_clamped - min_std))).item()
    position = action * confidence_scale
    return position


def compute_simple_reward(
    position_prev: float,
    position_curr: float,
    target: float,
    taker_fee: float = 0.0005,
    epsilon: float = 1e-8
) -> tuple[float, float, float]:
    """Compute PnL-based reward with transaction costs."""
    gross_pnl = position_curr * target
    position_change = abs(position_curr - position_prev)
    tc = taker_fee * position_change if position_change > epsilon else 0.0
    reward = gross_pnl - tc
    return reward, gross_pnl, tc


def compute_transaction_cost(
    position_prev: float,
    position_curr: float,
    taker_fee: float = 0.0005,
    epsilon: float = 1e-8
) -> float:
    """Compute transaction cost for position change."""
    position_change = abs(position_curr - position_prev)
    return taker_fee * position_change if position_change > epsilon else 0.0


def compute_unrealized_pnl(
    position: float,
    target: float
) -> float:
    """Compute unrealized PnL from current position."""
    return position * target


def compute_directional_bonus(
    position: float,
    target: float,
    bonus_weight: float = 0.000002
) -> float:
    """Compute directional accuracy bonus for reward shaping."""
    if abs(position) < 1e-6:
        return 0.0

    if (position * target) > 0:
        return bonus_weight * abs(position)
    else:
        return -bonus_weight * abs(position)
