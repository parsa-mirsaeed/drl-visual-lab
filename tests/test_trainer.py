"""Tests for Trainer (integration tests)."""

import pytest
import numpy as np
from core.trainer import Trainer


@pytest.fixture(scope="module")
def trained():
    """Train for 100 episodes once — shared across tests in this module."""
    t = Trainer(n_episodes=100, seed=0)
    history = t.train(verbose=False)
    return t, history


class TestTrainingLoop:
    def test_rewards_history_length(self, trained):
        _, history = trained
        assert len(history["rewards"]) == 100

    def test_losses_history_length(self, trained):
        _, history = trained
        assert len(history["losses"]) == 100

    def test_epsilon_decreases(self, trained):
        _, history = trained
        assert history["epsilons"][-1] < history["epsilons"][0]

    def test_policy_snapshots_taken(self, trained):
        _, history = trained
        # 100 episodes / 50 interval = 2 snapshots
        assert len(history["policy_snapshots"]) == 2

    def test_snapshot_shapes(self, trained):
        _, history = trained
        for snap in history["policy_snapshots"]:
            assert snap.shape == (25,)

    def test_q_snapshots_shapes(self, trained):
        _, history = trained
        for snap in history["q_snapshots"]:
            assert snap.shape == (25, 4)

    def test_some_episodes_reach_goal(self, trained):
        _, history = trained
        # After 100 episodes of DQN there should be at least one reward > 0
        assert max(history["rewards"]) > 0


class TestPlayEpisode:
    def test_trajectory_not_empty(self, trained):
        trainer, _ = trained
        traj = trainer.play_episode()
        assert len(traj) > 0

    def test_trajectory_structure(self, trained):
        trainer, _ = trained
        traj = trainer.play_episode()
        for pos, action, reward, outcome in traj:
            assert len(pos) == 2
            assert 0 <= action < 4
            assert isinstance(reward, float)
            assert outcome in {"step", "goal", "trap", "timeout"}

    def test_last_step_is_terminal(self, trained):
        trainer, _ = trained
        traj = trainer.play_episode()
        _, _, _, outcome = traj[-1]
        assert outcome in {"goal", "trap", "timeout"}
