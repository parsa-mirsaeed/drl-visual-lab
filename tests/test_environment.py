"""Tests for GridWorld environment."""

import pytest
import numpy as np
from core.environment import GridWorld


@pytest.fixture
def env():
    return GridWorld(seed=0)


class TestReset:
    def test_returns_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (env.n_states,)

    def test_start_position(self, env):
        env.reset()
        assert env.pos == [0, 0]

    def test_obs_is_one_hot(self, env):
        obs = env.reset()
        assert obs.sum() == pytest.approx(1.0)
        assert obs[0] == 1.0

    def test_done_is_false_after_reset(self, env):
        env.reset()
        assert not env.done

    def test_steps_reset_to_zero(self, env):
        env.reset()
        env.step(1)
        env.reset()
        assert env.steps == 0


class TestStep:
    def test_move_down(self, env):
        env.reset()
        obs, reward, done, info = env.step(1)  # DOWN
        assert env.pos == [1, 0]
        assert obs[1 * env.GRID_SIZE + 0] == 1.0

    def test_move_right(self, env):
        env.reset()
        obs, reward, done, info = env.step(3)  # RIGHT
        assert env.pos == [0, 1]

    def test_wall_clamp_up(self, env):
        env.reset()
        env.step(0)  # UP from (0,0) → stays (0,0)
        assert env.pos == [0, 0]

    def test_wall_clamp_left(self, env):
        env.reset()
        env.step(2)  # LEFT from (0,0) → stays (0,0)
        assert env.pos == [0, 0]

    def test_step_reward(self, env):
        env.reset()
        _, reward, done, info = env.step(3)
        assert reward == pytest.approx(GridWorld.REWARD_STEP)
        assert not done
        assert info["outcome"] == "step"

    def test_trap_reward_and_done(self, env):
        env.reset()
        env.pos = [1, 0]
        _, reward, done, info = env.step(3)  # RIGHT → (1,1) which is a trap
        assert reward == pytest.approx(GridWorld.REWARD_TRAP)
        assert done
        assert info["outcome"] == "trap"

    def test_goal_reward_and_done(self, env):
        env.reset()
        env.pos = [4, 3]
        _, reward, done, info = env.step(3)  # RIGHT → (4,4) GOAL
        assert reward == pytest.approx(GridWorld.REWARD_GOAL)
        assert done
        assert info["outcome"] == "goal"

    def test_step_after_done_raises(self, env):
        env.reset()
        env.pos = [4, 3]
        env.step(3)  # reach goal → done
        with pytest.raises(AssertionError):
            env.step(0)

    def test_obs_one_hot_after_step(self, env):
        env.reset()
        obs, *_ = env.step(3)
        assert obs.sum() == pytest.approx(1.0)

    def test_max_steps_timeout(self, env):
        env.reset()
        env.steps = GridWorld.MAX_STEPS - 1
        _, reward, done, info = env.step(3)
        assert done
        assert info["outcome"] == "timeout"


class TestRender:
    def test_render_contains_agent(self, env):
        env.reset()
        s = env.render_ascii()
        assert "A" in s

    def test_render_contains_goal(self, env):
        env.reset()
        s = env.render_ascii()
        assert "G" in s

    def test_render_contains_trap(self, env):
        env.reset()
        s = env.render_ascii()
        assert "T" in s
