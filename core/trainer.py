"""Training loop — ties environment, agent, and replay buffer together."""

from __future__ import annotations
import numpy as np
from typing import List, Dict
from core.environment import GridWorld
from core.agent import DQNAgent
from core.replay_buffer import ReplayBuffer


class Trainer:
    """Orchestrates DQN training on GridWorld."""

    def __init__(
        self,
        n_episodes: int = 500,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        warmup_steps: int = 200,
        seed: int = 42,
    ):
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        self.env = GridWorld(seed=seed)
        self.agent = DQNAgent(
            state_dim=self.env.n_states,
            n_actions=self.env.n_actions,
            seed=seed,
        )
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=self.env.n_states,
            seed=seed,
        )

        # History for visualization
        self.rewards_history: List[float] = []
        self.losses_history: List[float] = []
        self.epsilon_history: List[float] = []
        self.policy_snapshots: List[np.ndarray] = []  # every 50 eps
        self.q_snapshots: List[np.ndarray] = []       # every 50 eps

    def train(self, verbose: bool = True) -> Dict[str, list]:
        """Run full training. Returns history dict."""
        total_steps = 0

        for ep in range(1, self.n_episodes + 1):
            state = self.env.reset()
            ep_reward = 0.0
            ep_losses = []

            while True:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                total_steps += 1

                if len(self.buffer) >= self.warmup_steps:
                    batch = self.buffer.sample(self.batch_size)
                    loss = self.agent.update(*batch)
                    ep_losses.append(loss)

                if done:
                    break

            self.rewards_history.append(ep_reward)
            self.losses_history.append(
                float(np.mean(ep_losses)) if ep_losses else 0.0
            )
            self.epsilon_history.append(self.agent.epsilon)

            if ep % 50 == 0:
                self.policy_snapshots.append(
                    self.agent.greedy_policy(self.env.n_states).copy()
                )
                self.q_snapshots.append(
                    self.agent.get_q_table(self.env.n_states).copy()
                )
                if verbose:
                    avg_r = np.mean(self.rewards_history[-50:])
                    print(
                        f"  Ep {ep:4d}/{self.n_episodes} | "
                        f"Avg Reward: {avg_r:6.2f} | "
                        f"ε: {self.agent.epsilon:.3f} | "
                        f"Buffer: {len(self.buffer)}"
                    )

        return {
            "rewards": self.rewards_history,
            "losses": self.losses_history,
            "epsilons": self.epsilon_history,
            "policy_snapshots": self.policy_snapshots,
            "q_snapshots": self.q_snapshots,
        }

    def play_episode(self) -> List[tuple]:
        """Run one greedy episode. Returns list of (pos, action, reward)."""
        state = self.env.reset()
        trajectory = []
        while True:
            action = self.agent.select_action(state, greedy=True)
            pos = tuple(self.env.pos)
            next_state, reward, done, info = self.env.step(action)
            trajectory.append((pos, action, reward, info["outcome"]))
            state = next_state
            if done:
                break
        return trajectory
