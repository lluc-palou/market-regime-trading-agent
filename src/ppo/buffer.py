"""Trajectory buffer and state management."""

import torch
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition in trajectory."""
    codebooks: torch.Tensor      # (window_size,)
    features: torch.Tensor        # (window_size, n_features)
    timestamps: torch.Tensor      # (window_size,)
    action: float
    log_prob: float
    reward: float
    value: float
    done: bool


class StateBuffer:
    """Rolling window buffer for state construction."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.codebooks = deque(maxlen=window_size)
        self.features = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add(self, codebook: int, features: torch.Tensor, timestamp: float):
        """Add new sample to buffer."""
        self.codebooks.append(codebook)
        self.features.append(features)
        self.timestamps.append(timestamp)
    
    def get_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Return current state as tensors.
        
        Returns:
            None if buffer not full, otherwise dict with:
                - codebooks: (window_size,)
                - features: (window_size, n_features)
                - timestamps: (window_size,)
        """
        if len(self.codebooks) < self.window_size:
            return None
        
        return {
            'codebooks': torch.tensor(list(self.codebooks), dtype=torch.long),
            'features': torch.stack(list(self.features)),
            'timestamps': torch.tensor(list(self.timestamps), dtype=torch.float32)
        }
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples."""
        return len(self.codebooks) >= self.window_size
    
    def reset(self):
        """Clear buffer."""
        self.codebooks.clear()
        self.features.clear()
        self.timestamps.clear()


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
                - features: (T, window_size, n_features)
                - timestamps: (T, window_size)
                - actions: (T,)
                - log_probs: (T,)
                - rewards: (T,)
                - values: (T,)
                - dones: (T,)
        """
        if len(self.transitions) == 0:
            raise ValueError("Buffer is empty")
        
        return {
            'codebooks': torch.stack([t.codebooks for t in self.transitions]).to(device),
            'features': torch.stack([t.features for t in self.transitions]).to(device),
            'timestamps': torch.stack([t.timestamps for t in self.transitions]).to(device),
            'actions': torch.tensor([t.action for t in self.transitions], dtype=torch.float32).to(device),
            'log_probs': torch.tensor([t.log_prob for t in self.transitions], dtype=torch.float32).to(device),
            'rewards': torch.tensor([t.reward for t in self.transitions], dtype=torch.float32).to(device),
            'values': torch.tensor([t.value for t in self.transitions], dtype=torch.float32).to(device),
            'dones': torch.tensor([t.done for t in self.transitions], dtype=torch.float32).to(device)
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
        self.cumulative_tc = 0.0
        self.trade_count = 0
        self.step_count = 0
        
        # Episode metrics
        self.max_pnl = float('-inf')
        self.min_pnl = float('inf')
        self.max_position = 0.0
        self.total_reward = 0.0
        
        # Current unrealized
        self.unrealized_pnl = 0.0
    
    def update(
        self,
        action: float,
        log_return: float,
        transaction_cost: float,
        reward: float,
        unrealized_pnl: float
    ):
        """
        Update agent state after taking action.
        
        Args:
            action: New position taken
            log_return: Return that occurred this timestep
            transaction_cost: TC paid for position change
            reward: Reward received
            unrealized_pnl: Expected PnL from new position
        """
        # Realized PnL from previous position
        realized_this_step = self.current_position * log_return
        self.cumulative_realized_pnl += realized_this_step
        
        # Transaction costs
        self.cumulative_tc += transaction_cost
        
        # Track trades
        if abs(action - self.current_position) > 0.01:
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
        self.total_reward += reward
        self.step_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics."""
        net_pnl = self.cumulative_realized_pnl - self.cumulative_tc
        total_pnl = net_pnl + self.unrealized_pnl
        
        return {
            'net_realized_pnl': net_pnl,
            'cumulative_tc': self.cumulative_tc,
            'total_pnl': total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'trade_count': self.trade_count,
            'trade_frequency': self.trade_count / max(self.step_count, 1),
            'max_drawdown': self.max_pnl - self.min_pnl,
            'max_position': self.max_position,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.step_count, 1)
        }
    
    def reset(self):
        """Reset for new episode."""
        self.__init__()