"""Tests for ReplayBuffer."""

import pytest
import numpy as np
from core.replay_buffer import ReplayBuffer


STATE_DIM = 25


@pytest.fixture
def buf():
    return ReplayBuffer(capacity=100, state_dim=STATE_DIM, seed=0)


def _dummy_transition(i: int = 0):
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s[i % STATE_DIM] = 1.0
    a = i % 4
    r = float(i)
    s_ = np.roll(s, 1)
    d = (i % 10 == 0)
    return s, a, r, s_, d


class TestPushAndLen:
    def test_empty_buffer(self, buf):
        assert len(buf) == 0

    def test_push_increments_size(self, buf):
        buf.push(*_dummy_transition(0))
        assert len(buf) == 1

    def test_push_multiple(self, buf):
        for i in range(50):
            buf.push(*_dummy_transition(i))
        assert len(buf) == 50

    def test_capacity_overflow_wraps(self, buf):
        # Push more than capacity
        for i in range(120):
            buf.push(*_dummy_transition(i))
        assert len(buf) == 100  # capped at capacity

    def test_fill_ratio_full(self, buf):
        for i in range(100):
            buf.push(*_dummy_transition(i))
        assert buf.fill_ratio == pytest.approx(1.0)

    def test_fill_ratio_half(self, buf):
        for i in range(50):
            buf.push(*_dummy_transition(i))
        assert buf.fill_ratio == pytest.approx(0.5)


class TestSample:
    def test_sample_shapes(self, buf):
        for i in range(64):
            buf.push(*_dummy_transition(i))
        s, a, r, s_, d = buf.sample(32)
        assert s.shape == (32, STATE_DIM)
        assert a.shape == (32,)
        assert r.shape == (32,)
        assert s_.shape == (32, STATE_DIM)
        assert d.shape == (32,)

    def test_sample_raises_if_too_small(self, buf):
        buf.push(*_dummy_transition(0))
        with pytest.raises(AssertionError):
            buf.sample(32)

    def test_sample_no_repeat_indices(self, buf):
        """With replace=False, sampled indices must be unique."""
        for i in range(64):
            buf.push(*_dummy_transition(i))
        s, a, r, s_, d = buf.sample(32)
        # Rewards are unique (each is float(i)), so if no repeats, all unique
        assert len(np.unique(r)) == 32

    def test_done_dtype_float(self, buf):
        buf.push(*_dummy_transition(0))
        for i in range(63):
            buf.push(*_dummy_transition(i + 1))
        _, _, _, _, d = buf.sample(32)
        assert d.dtype == np.float32
