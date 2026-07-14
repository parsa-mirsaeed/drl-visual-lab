"""Experience Replay Buffer — stores (s, a, r, s', done) transitions.

Key idea: break temporal correlations by sampling random mini-batches
instead of training on consecutive steps.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class ReplayBuffer:
    """Fixed-capacity circular replay buffer."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 42):
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self._ptr = 0
        self._size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        i = self._ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Sample a random mini-batch of transitions."""
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} samples, need {batch_size}."
        )
        idx = self.rng.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def fill_ratio(self) -> float:
        """Fraction of buffer that is filled (0.0 → 1.0)."""
        return self._size / self.capacity
