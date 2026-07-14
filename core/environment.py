"""GridWorld environment — a minimal MDP for DRL experiments.

Grid layout (5x5 default):
  S = start, G = goal, T = trap, . = free cell

  S . . . .
  . T . T .
  . . . . .
  . T . T .
  . . . . G

Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
Reward:  +10 goal, -5 trap, -0.1 step penalty
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any


ACTION_DELTAS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class GridWorld:
    """5×5 GridWorld MDP environment."""

    GRID_SIZE = 5
    START = (0, 0)
    GOAL = (4, 4)
    TRAPS = {(1, 1), (1, 3), (3, 1), (3, 3)}

    REWARD_GOAL = 10.0
    REWARD_TRAP = -5.0
    REWARD_STEP = -0.1
    MAX_STEPS = 200

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_states = self.GRID_SIZE * self.GRID_SIZE
        self.n_actions = 4
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset to start state. Returns one-hot state vector."""
        self.pos = list(self.START)
        self.steps = 0
        self.done = False
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action. Returns (next_obs, reward, done, info)."""
        assert not self.done, "Episode finished — call reset() first."
        assert 0 <= action < self.n_actions

        dr, dc = ACTION_DELTAS[action]
        nr = np.clip(self.pos[0] + dr, 0, self.GRID_SIZE - 1)
        nc = np.clip(self.pos[1] + dc, 0, self.GRID_SIZE - 1)
        self.pos = [int(nr), int(nc)]
        self.steps += 1

        pos_tuple = tuple(self.pos)
        if pos_tuple == self.GOAL:
            reward = self.REWARD_GOAL
            self.done = True
            info = {"outcome": "goal"}
        elif pos_tuple in self.TRAPS:
            reward = self.REWARD_TRAP
            self.done = True
            info = {"outcome": "trap"}
        elif self.steps >= self.MAX_STEPS:
            reward = self.REWARD_STEP
            self.done = True
            info = {"outcome": "timeout"}
        else:
            reward = self.REWARD_STEP
            info = {"outcome": "step"}

        return self._obs(), reward, self.done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        """One-hot encode current position into a state vector."""
        idx = self.pos[0] * self.GRID_SIZE + self.pos[1]
        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[idx] = 1.0
        return obs

    def state_index(self) -> int:
        return self.pos[0] * self.GRID_SIZE + self.pos[1]

    def render_ascii(self) -> str:
        """Return ASCII grid string for terminal display."""
        lines = []
        for r in range(self.GRID_SIZE):
            row = []
            for c in range(self.GRID_SIZE):
                t = (r, c)
                if [r, c] == self.pos:
                    row.append("A")
                elif t == self.GOAL:
                    row.append("G")
                elif t in self.TRAPS:
                    row.append("T")
                elif t == self.START:
                    row.append("S")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    @property
    def grid_size(self) -> int:
        return self.GRID_SIZE
