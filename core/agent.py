"""DQN Agent — Deep Q-Network from scratch using PyTorch.

Architecture:
  state (one-hot) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(n_actions)

Key tricks:
  - Target network (soft update every N steps)
  - Epsilon-greedy exploration with decay
  - Experience replay (mini-batch gradient descent)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class QNetwork(nn.Module):
    """Fully-connected Q-value approximator."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Deep Q-Network agent."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self._step_count = 0
        self.rng = np.random.default_rng(seed)

        self.q_net = QNetwork(state_dim, n_actions)
        self.target_net = QNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.q_net(s)
        return int(q_vals.argmax().item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one gradient update step. Returns scalar loss."""
        s = torch.FloatTensor(states)
        a = torch.LongTensor(actions)
        r = torch.FloatTensor(rewards)
        s_ = torch.FloatTensor(next_states)
        d = torch.FloatTensor(dones)

        # Current Q-values
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            max_q_next = self.target_net(s_).max(1)[0]
            targets = r + self.gamma * max_q_next * (1 - d)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def get_q_table(self, n_states: int) -> np.ndarray:
        """Return Q-values for all states. Shape: (n_states, n_actions)."""
        eye = torch.eye(n_states)
        with torch.no_grad():
            q = self.q_net(eye).numpy()
        return q

    def greedy_policy(self, n_states: int) -> np.ndarray:
        """Return best action index for each state."""
        return self.get_q_table(n_states).argmax(axis=1)
