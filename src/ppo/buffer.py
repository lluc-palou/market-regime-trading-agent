"""Trajectory buffer and state management."""

import torch
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition in trajectory."""
    codebooks: torch.Tensor                    # (window_size,)
    features: Optional[torch.Tensor]           # (window_size, n_features) or None for codebook-only
    timestamps: torch.Tensor                    # (window_size,)
    action: float
    log_prob: float
    reward: float
    value: float
    done: bool


class StateBuffer:
    """Rolling window buffer for state construction with tensor caching."""

    def __init__(self, window_size: int, use_pinned_memory: bool = True):
        self.window_size = window_size
        self.use_pinned_memory = use_pinned_memory

        # Store as lists for fast append
        self.codebooks = deque(maxlen=window_size)
        self.features = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Cache for avoiding tensor recreation
        self._cached_state = None
        self._cache_valid = False

    def add(self, codebook: int, features: torch.Tensor, timestamp: float):
        """Add new sample to buffer."""
        self.codebooks.append(codebook)
        self.features.append(features)
        self.timestamps.append(timestamp)
        # Invalidate cache when new data added
        self._cache_valid = False

    def get_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Return current state as tensors with caching.

        Returns:
            None if buffer not full, otherwise dict with:
                - codebooks: (window_size,)
                - features: (window_size, n_features) or None for codebook-only experiments
                - timestamps: (window_size,)
        """
        if len(self.codebooks) < self.window_size:
            return None

        # Return cached state if valid
        if self._cache_valid and self._cached_state is not None:
            return self._cached_state

        # Create new tensors with optional pinned memory
        if self.use_pinned_memory:
            codebooks_tensor = torch.tensor(
                list(self.codebooks), dtype=torch.long
            ).pin_memory()
            # Handle None features for codebook-only experiments
            if self.features[0] is not None:
                features_tensor = torch.stack(list(self.features)).pin_memory()
            else:
                features_tensor = None
            timestamps_tensor = torch.tensor(
                list(self.timestamps), dtype=torch.float32
            ).pin_memory()
        else:
            codebooks_tensor = torch.tensor(list(self.codebooks), dtype=torch.long)
            # Handle None features for codebook-only experiments
            if self.features[0] is not None:
                features_tensor = torch.stack(list(self.features))
            else:
                features_tensor = None
            timestamps_tensor = torch.tensor(list(self.timestamps), dtype=torch.float32)

        self._cached_state = {
            'codebooks': codebooks_tensor,
            'features': features_tensor,
            'timestamps': timestamps_tensor
        }
        self._cache_valid = True

        return self._cached_state
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples."""
        return len(self.codebooks) >= self.window_size
    
    def reset(self):
        """Clear buffer and cache."""
        self.codebooks.clear()
        self.features.clear()
        self.timestamps.clear()
        self._cached_state = None
        self._cache_valid = False


class TrajectoryBuffer:
    """Buffer for storing trajectories for PPO updates."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.transitions: List[Transition] = []
    
    def add(self, transition: Transition):
        """Add transition to buffer."""
        self.transitions.append(transition)
    
    def get_batch(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Return all transitions as batched tensors.

        Returns:
            Dictionary with batched tensors:
                - codebooks: (T, window_size)
                - features: (T, window_size, n_features) or None for codebook-only experiments
                - timestamps: (T, window_size)
                - actions: (T,)
                - log_probs: (T,)
                - rewards: (T,)
                - values: (T,)
                - dones: (T,)
        """
        if len(self.transitions) == 0:
            raise ValueError("Buffer is empty")

        # Handle None features for codebook-only experiments
        if self.transitions[0].features is not None:
            features_batch = torch.stack([t.features for t in self.transitions]).to(device, non_blocking=True)
        else:
            features_batch = None

        # Use non-blocking transfers for better performance (works with pinned memory)
        return {
            'codebooks': torch.stack([t.codebooks for t in self.transitions]).to(device, non_blocking=True),
            'features': features_batch,
            'timestamps': torch.stack([t.timestamps for t in self.transitions]).to(device, non_blocking=True),
            'actions': torch.tensor([t.action for t in self.transitions], dtype=torch.float32).to(device, non_blocking=True),
            'log_probs': torch.tensor([t.log_prob for t in self.transitions], dtype=torch.float32).to(device, non_blocking=True),
            'rewards': torch.tensor([t.reward for t in self.transitions], dtype=torch.float32).to(device, non_blocking=True),
            'values': torch.tensor([t.value for t in self.transitions], dtype=torch.float32).to(device, non_blocking=True),
            'dones': torch.tensor([t.done for t in self.transitions], dtype=torch.float32).to(device, non_blocking=True)
        }
    
    def clear(self):
        """Clear all transitions."""
        self.transitions.clear()
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def is_full(self) -> bool:
        """Check if buffer reached capacity."""
        return len(self.transitions) >= self.capacity


class AgentState:
    """Tracks agent trading state during episode."""
    
    def __init__(self):
        # Trading state
        self.current_position = 0.0
        self.cumulative_realized_pnl = 0.0
        self.cumulative_gross_pnl = 0.0  # NEW: Track gross PnL separately
        self.cumulative_tc = 0.0
        self.cumulative_position_change = 0.0  # NEW: Track total position changes for TC calculations
        self.trade_count = 0
        self.step_count = 0

        # Episode metrics
        self.max_pnl = float('-inf')
        self.min_pnl = float('inf')
        self.max_position = 0.0
        self.total_reward = 0.0  # Learning signal (includes directional bonus)

        # Trading returns for different fee scenarios (all exclude directional bonus)
        # Volatility-normalized (for training rewards)
        self.total_trading_return_raw = 0.0  # Baseline: Buy-and-hold (no position sizing, no fees)
        self.total_trading_return_taker = 0.0  # Taker fee 5 bps (market orders with agent sizing)
        self.total_trading_return_maker_neutral = 0.0  # Maker fee 0 bps (limit orders with agent sizing)
        self.total_trading_return_maker_rebate = 0.0  # Maker rebate -2.5 bps (limit orders with agent sizing)

        # Legacy field for backward compatibility (points to taker)
        self.total_trading_return = 0.0  # Same as total_trading_return_taker

        # Raw log returns for different fee scenarios (for Sharpe analysis - NOT volatility normalized)
        self.cumulative_raw_return_buyhold = 0.0  # Buy-and-hold baseline
        self.cumulative_raw_return_taker = 0.0  # Taker fee 10 bps
        self.cumulative_raw_return_maker_neutral = 0.0  # Maker fee 0 bps
        self.cumulative_raw_return_maker_rebate = 0.0  # Maker rebate -2.5 bps

        # Position tracking
        self.sum_abs_position = 0.0  # For mean absolute position

        # Action std tracking (policy uncertainty/confidence)
        self.sum_action_std = 0.0
        self.min_action_std = float('inf')
        self.max_action_std = float('-inf')

        # Current unrealized
        self.unrealized_pnl = 0.0
    
    def update(
        self,
        action: float,
        log_return: float,
        transaction_cost: float,
        reward: float,
        unrealized_pnl: float,
        gross_pnl: float = 0.0,  # Gross PnL from current timestep
        action_std: float = 0.0,  # Policy std (agent's uncertainty/confidence)
        trading_return_raw: float = 0.0,  # Volatility-normalized: Buy-and-hold baseline
        trading_return_taker: float = 0.0,  # Volatility-normalized: Taker 10 bps
        trading_return_maker_neutral: float = 0.0,  # Volatility-normalized: Maker 0 bps
        trading_return_maker_rebate: float = 0.0,  # Volatility-normalized: Maker -2.5 bps
        raw_log_return_buyhold: float = 0.0,  # Raw log return: Buy-and-hold baseline
        raw_log_return_taker: float = 0.0,  # Raw log return: Taker 10 bps
        raw_log_return_maker_neutral: float = 0.0,  # Raw log return: Maker 0 bps
        raw_log_return_maker_rebate: float = 0.0  # Raw log return: Maker -2.5 bps
    ):
        """
        Update agent state after taking action.

        Args:
            action: New position taken (policy-based)
            log_return: Return that occurred this timestep
            transaction_cost: TC paid for position change
            reward: Reward received (includes directional bonus for learning)
            unrealized_pnl: Expected PnL from new position
            gross_pnl: Gross PnL from current position (before TC)
            action_std: Policy standard deviation (agent's learned uncertainty)
            trading_return_*: Volatility-normalized returns (for training rewards)
            raw_log_return_*: Raw log returns NOT normalized (for Sharpe analysis)
        """
        # Realized PnL from previous position
        realized_this_step = self.current_position * log_return
        self.cumulative_realized_pnl += realized_this_step

        # Gross PnL tracking
        self.cumulative_gross_pnl += gross_pnl

        # Transaction costs
        self.cumulative_tc += transaction_cost

        # Track position changes (for computing TC in different scenarios)
        position_change = abs(action - self.current_position)
        self.cumulative_position_change += position_change

        # Track trades
        if position_change > 0.01:
            self.trade_count += 1

        # Update position
        self.current_position = action

        # Unrealized PnL
        self.unrealized_pnl = unrealized_pnl

        # Track metrics
        net_pnl = self.cumulative_realized_pnl - self.cumulative_tc
        self.max_pnl = max(self.max_pnl, net_pnl)
        self.min_pnl = min(self.min_pnl, net_pnl)
        self.max_position = max(self.max_position, abs(action))
        self.sum_abs_position += abs(action)
        self.total_reward += reward  # Learning signal (with directional bonus)

        # Track volatility-normalized trading returns (for training rewards)
        self.total_trading_return_raw += trading_return_raw  # Baseline: no fees
        self.total_trading_return_taker += trading_return_taker  # Taker 10 bps (market orders)
        self.total_trading_return_maker_neutral += trading_return_maker_neutral  # Maker 0 bps (limit orders)
        self.total_trading_return_maker_rebate += trading_return_maker_rebate  # Maker -2.5 bps (limit orders)
        self.total_trading_return = trading_return_taker  # Legacy field (same as taker)

        # Track raw log returns (for Sharpe analysis - NOT volatility normalized)
        self.cumulative_raw_return_buyhold += raw_log_return_buyhold
        self.cumulative_raw_return_taker += raw_log_return_taker
        self.cumulative_raw_return_maker_neutral += raw_log_return_maker_neutral
        self.cumulative_raw_return_maker_rebate += raw_log_return_maker_rebate

        self.step_count += 1

        # Track action std (policy uncertainty)
        self.sum_action_std += abs(action_std)
        self.min_action_std = min(self.min_action_std, abs(action_std))
        self.max_action_std = max(self.max_action_std, abs(action_std))
    
    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics."""
        net_pnl = self.cumulative_realized_pnl - self.cumulative_tc
        total_pnl = net_pnl + self.unrealized_pnl

        # Gross PnL metrics (realized PnL before TC)
        avg_gross_pnl_per_trade = self.cumulative_realized_pnl / max(self.trade_count, 1)
        avg_tc_per_trade = self.cumulative_tc / max(self.trade_count, 1)
        avg_position_change_per_trade = self.cumulative_position_change / max(self.trade_count, 1)
        pnl_to_cost_ratio = self.cumulative_realized_pnl / self.cumulative_tc if self.cumulative_tc > 1e-8 else 0.0

        # Position metrics
        mean_abs_position = self.sum_abs_position / max(self.step_count, 1)

        # Action std metrics (policy uncertainty/confidence)
        mean_action_std = self.sum_action_std / max(self.step_count, 1)

        return {
            'net_realized_pnl': net_pnl,
            'cumulative_tc': self.cumulative_tc,
            'total_pnl': total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'trade_count': self.trade_count,
            'trade_frequency': self.trade_count / max(self.step_count, 1),
            'max_drawdown': self.max_pnl - self.min_pnl,
            'max_position': self.max_position,
            'mean_abs_position': mean_abs_position,
            'total_reward': self.total_reward,  # Learning signal (with directional bonus)
            'avg_reward': self.total_reward / max(self.step_count, 1),

            # Volatility-normalized trading returns (for training rewards, exclude directional bonus)
            'total_trading_return_raw': self.total_trading_return_raw,  # Baseline: buy-and-hold
            'total_trading_return_taker': self.total_trading_return_taker,  # Taker 10 bps (agent sizing)
            'total_trading_return_maker_neutral': self.total_trading_return_maker_neutral,  # Maker 0 bps (agent sizing)
            'total_trading_return_maker_rebate': self.total_trading_return_maker_rebate,  # Maker -2.5 bps (agent sizing)
            'total_trading_return': self.total_trading_return,  # Legacy (same as taker)
            'avg_trading_return': self.total_trading_return / max(self.step_count, 1),

            # Raw log returns (for Sharpe analysis - NOT volatility normalized)
            'cumulative_raw_return_buyhold': self.cumulative_raw_return_buyhold,
            'cumulative_raw_return_taker': self.cumulative_raw_return_taker,
            'cumulative_raw_return_maker_neutral': self.cumulative_raw_return_maker_neutral,
            'cumulative_raw_return_maker_rebate': self.cumulative_raw_return_maker_rebate,
            # Gross PnL metrics
            'cumulative_gross_pnl': self.cumulative_gross_pnl,
            'avg_gross_pnl_per_trade': avg_gross_pnl_per_trade,
            'avg_tc_per_trade': avg_tc_per_trade,
            'avg_position_change_per_trade': avg_position_change_per_trade,
            'pnl_to_cost_ratio': pnl_to_cost_ratio,
            # Action std metrics (policy uncertainty - low std = high confidence)
            'mean_action_std': mean_action_std,
            'min_action_std': self.min_action_std,
            'max_action_std': self.max_action_std
        }
    
    def reset(self):
        """Reset for new episode."""
        self.__init__()