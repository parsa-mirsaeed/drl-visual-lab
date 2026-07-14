"""Tests for DQNAgent."""

import pytest
import numpy as np
import torch
from core.agent import DQNAgent, QNetwork


STATE_DIM = 25
N_ACTIONS = 4


@pytest.fixture
def agent():
    return DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, seed=0)


@pytest.fixture
def state():
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s[0] = 1.0
    return s


class TestQNetwork:
    def test_output_shape(self):
        net = QNetwork(STATE_DIM, N_ACTIONS)
        x = torch.zeros(1, STATE_DIM)
        out = net(x)
        assert out.shape == (1, N_ACTIONS)

    def test_batch_output_shape(self):
        net = QNetwork(STATE_DIM, N_ACTIONS)
        x = torch.zeros(16, STATE_DIM)
        out = net(x)
        assert out.shape == (16, N_ACTIONS)

    def test_gradient_flows(self):
        net = QNetwork(STATE_DIM, N_ACTIONS)
        x = torch.randn(4, STATE_DIM)
        out = net(x).sum()
        out.backward()
        for p in net.parameters():
            assert p.grad is not None


class TestActionSelection:
    def test_action_in_valid_range(self, agent, state):
        for _ in range(50):
            a = agent.select_action(state)
            assert 0 <= a < N_ACTIONS

    def test_greedy_action_is_deterministic(self, agent, state):
        actions = {agent.select_action(state, greedy=True) for _ in range(10)}
        assert len(actions) == 1  # always same action

    def test_epsilon_one_always_random(self, agent, state):
        agent.epsilon = 1.0
        # With epsilon=1.0 force random; over 100 tries should see multiple actions
        actions = {agent.select_action(state) for _ in range(100)}
        assert len(actions) > 1

    def test_epsilon_zero_is_greedy(self, agent, state):
        agent.epsilon = 0.0
        actions = {agent.select_action(state) for _ in range(20)}
        assert len(actions) == 1


class TestUpdate:
    def _make_batch(self, n=32):
        states = np.eye(STATE_DIM, dtype=np.float32)[:n]
        actions = np.zeros(n, dtype=np.int64)
        rewards = np.random.randn(n).astype(np.float32)
        next_states = np.eye(STATE_DIM, dtype=np.float32)[:n]
        dones = np.zeros(n, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def test_update_returns_scalar_loss(self, agent):
        loss = agent.update(*self._make_batch())
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_epsilon_decays_after_update(self, agent):
        start_eps = agent.epsilon
        agent.update(*self._make_batch())
        assert agent.epsilon < start_eps

    def test_epsilon_never_below_min(self, agent):
        agent.epsilon = agent.epsilon_end
        agent.update(*self._make_batch())
        assert agent.epsilon >= agent.epsilon_end

    def test_target_network_updates(self, agent):
        agent.target_update_freq = 1
        batch = self._make_batch()
        agent.update(*batch)
        # After update, q_net and target_net params should be identical
        for p1, p2 in zip(
            agent.q_net.parameters(), agent.target_net.parameters()
        ):
            assert torch.allclose(p1, p2)


class TestQTable:
    def test_q_table_shape(self, agent):
        q = agent.get_q_table(STATE_DIM)
        assert q.shape == (STATE_DIM, N_ACTIONS)

    def test_greedy_policy_shape(self, agent):
        p = agent.greedy_policy(STATE_DIM)
        assert p.shape == (STATE_DIM,)

    def test_greedy_policy_valid_actions(self, agent):
        p = agent.greedy_policy(STATE_DIM)
        assert all(0 <= a < N_ACTIONS for a in p)
