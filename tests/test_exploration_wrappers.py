"""
Tests for the count-based exploration and episodic state novelty wrappers.

Uses a minimal mock gymnasium environment with Dict observations to verify
wrapper behavior without requiring PyBoy or any game ROM.
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import sys
import os

# Add parent directory to path so we can import the wrappers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "onefiveone"))

from wrappers.count_based_exploration import CountBasedExplorationWrapper
from wrappers.episodic_novelty import EpisodicStateNoveltyWrapper


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

class MockGridEnv(gym.Env):
    """
    A trivial grid environment with Dict observations for testing wrappers.

    The agent lives on a 10x10 grid. Actions move up/down/left/right.
    Extrinsic reward is always 0.0 so we can isolate wrapper bonuses.
    The observation includes a "state_value" that changes independently
    of position, simulating internal game state changes (items, flags, etc).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = Dict({
            "coords": Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "map_id": Discrete(4),
            "state_value": Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        })
        self.action_space = Discrete(4)  # up, down, left, right
        self.x = 5
        self.y = 5
        self.map_id = 0
        self.state_value = np.zeros(8, dtype=np.float32)

    def _get_obs(self):
        return {
            "coords": np.array([self.x / 9.0, self.y / 9.0], dtype=np.float32),
            "map_id": self.map_id,
            "state_value": self.state_value.copy(),
        }

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.x = 5
        self.y = 5
        self.map_id = 0
        self.state_value = np.zeros(8, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        if action == 0 and self.y < 9:
            self.y += 1
        elif action == 1 and self.y > 0:
            self.y -= 1
        elif action == 2 and self.x > 0:
            self.x -= 1
        elif action == 3 and self.x < 9:
            self.x += 1

        return self._get_obs(), 0.0, False, False, {}

    def set_position(self, x, y, map_id=0):
        """Helper to teleport agent for testing."""
        self.x = x
        self.y = y
        self.map_id = map_id

    def set_state_value(self, values):
        """Helper to change internal state for testing."""
        self.state_value = np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# State key/hash functions for MockGridEnv
# ---------------------------------------------------------------------------

def mock_state_key_fn(obs):
    """Position-only key."""
    x = int(obs["coords"][0] * 9)
    y = int(obs["coords"][1] * 9)
    return (obs["map_id"], x, y)


def mock_state_hash_fn(obs):
    """Full state hash: position + internal state."""
    x = int(obs["coords"][0] * 9)
    y = int(obs["coords"][1] * 9)
    sv = tuple((obs["state_value"] * 255).astype(np.uint8))
    return (obs["map_id"], x, y, sv)


# =========================================================================
# CountBasedExplorationWrapper tests
# =========================================================================

class TestCountBasedExploration:
    def _make_wrapped(self, **kwargs):
        env = MockGridEnv()
        return CountBasedExplorationWrapper(
            env, state_key_fn=mock_state_key_fn, **kwargs
        )

    def test_first_visit_gives_bonus(self):
        env = self._make_wrapped(scale=1.0, exponent=0.5)
        env.reset()
        obs, reward, _, _, info = env.step(0)  # move up
        # First visit: bonus = 1.0 / (1 + 1^0.5) = 0.5
        assert reward > 0.0
        assert info["state_visit_count"] == 1.0
        assert info["exploration_bonus"] == pytest.approx(0.5, abs=0.01)

    def test_repeated_visits_decay_bonus(self):
        env = self._make_wrapped(scale=1.0, exponent=0.5)
        env.reset()

        # Visit the same tile multiple times by going up then down
        bonuses = []
        for _ in range(5):
            _, r1, _, _, info1 = env.step(0)  # up to (5,6)
            bonuses.append(info1["exploration_bonus"])
            env.step(1)  # back to (5,5)

        # Each revisit of (5,6) should give a smaller bonus
        for i in range(1, len(bonuses)):
            assert bonuses[i] < bonuses[i - 1], (
                f"Bonus should decay: visit {i} ({bonuses[i]}) >= visit {i-1} ({bonuses[i-1]})"
            )

    def test_unique_states_count(self):
        env = self._make_wrapped(scale=1.0)
        env.reset()

        # Move to 3 distinct tiles
        env.step(0)  # (5,6)
        env.step(0)  # (5,7)
        _, _, _, _, info = env.step(0)  # (5,8)

        # Starting tile (5,5) wasn't counted in step (only on reset obs),
        # so we have 3 unique states from the 3 steps
        assert info["unique_states_visited"] == 3

    def test_scale_parameter(self):
        env_low = self._make_wrapped(scale=0.1)
        env_high = self._make_wrapped(scale=10.0)
        env_low.reset()
        env_high.reset()

        _, r_low, _, _, _ = env_low.step(0)
        _, r_high, _, _, _ = env_high.step(0)

        assert r_high > r_low

    def test_decay_on_reset(self):
        env = self._make_wrapped(scale=1.0, decay_on_reset=0.0)
        env.reset()

        # Visit a tile
        env.step(0)
        assert len(env.visit_counts) > 0

        # Reset with full decay should clear everything
        env.reset()
        assert len(env.visit_counts) == 0

    def test_partial_decay_on_reset(self):
        env = self._make_wrapped(scale=1.0, decay_on_reset=0.5)
        env.reset()

        # Visit a tile many times so count is high
        for _ in range(20):
            env.step(0)  # up
            env.step(1)  # down

        counts_before = dict(env.visit_counts)
        env.reset()
        counts_after = dict(env.visit_counts)

        # Counts should be roughly halved (some may be pruned if below 0.5)
        for key in counts_after:
            assert counts_after[key] < counts_before.get(key, float("inf"))

    def test_no_decay_preserves_counts(self):
        env = self._make_wrapped(scale=1.0, decay_on_reset=1.0)
        env.reset()

        env.step(0)
        counts_before = dict(env.visit_counts)
        env.reset()
        counts_after = dict(env.visit_counts)

        assert counts_before == counts_after

    def test_max_states_pruning(self):
        env = self._make_wrapped(scale=1.0, max_states=3)
        env.reset()

        # Visit 5 different tiles
        for _ in range(5):
            env.step(0)  # move up through (5,6), (5,7), (5,8), (5,9)...

        # Should be capped at 3
        assert len(env.visit_counts) <= 3

    def test_extrinsic_reward_preserved(self):
        """Wrapper should add bonus ON TOP of extrinsic reward."""
        env = self._make_wrapped(scale=1.0)
        env.reset()

        # The mock env gives 0.0 extrinsic reward
        _, reward, _, _, info = env.step(0)
        assert info["extrinsic_reward"] == 0.0
        assert reward == info["exploration_bonus"]  # total = extrinsic + bonus

    def test_observation_unchanged(self):
        """Wrapper should not modify observations."""
        base_env = MockGridEnv()
        wrapped_env = CountBasedExplorationWrapper(
            MockGridEnv(), state_key_fn=mock_state_key_fn
        )
        base_obs, _ = base_env.reset()
        wrapped_obs, _ = wrapped_env.reset()

        for key in base_obs:
            if isinstance(base_obs[key], np.ndarray):
                np.testing.assert_array_equal(base_obs[key], wrapped_obs[key])
            else:
                assert base_obs[key] == wrapped_obs[key]

    def test_get_stats(self):
        env = self._make_wrapped(scale=1.0)
        env.reset()
        env.step(0)
        env.step(0)
        stats = env.get_stats()
        assert "unique_states" in stats
        assert "total_visits" in stats
        assert "mean_visits" in stats
        assert stats["unique_states"] >= 1


# =========================================================================
# EpisodicStateNoveltyWrapper tests
# =========================================================================

class TestEpisodicNovelty:
    def _make_wrapped(self, **kwargs):
        env = MockGridEnv()
        return EpisodicStateNoveltyWrapper(
            env, state_hash_fn=mock_state_hash_fn, **kwargs
        )

    def test_first_visit_gives_bonus(self):
        env = self._make_wrapped(novelty_bonus=1.0)
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward > 0.0
        assert info["is_novel_state"] is True

    def test_exact_revisit_no_bonus(self):
        """Revisiting the exact same state should give no bonus."""
        env = self._make_wrapped(novelty_bonus=1.0)
        env.reset()

        env.step(0)  # move to (5,6) — novel
        env.step(1)  # move back to (5,5)
        _, reward, _, _, info = env.step(0)  # move to (5,6) again — same state

        assert info["is_novel_state"] is False
        assert info["novelty_bonus"] == 0.0

    def test_novel_revisit_with_changed_state(self):
        """Same position but different internal state should give bonus."""
        env = self._make_wrapped(
            novelty_bonus=1.0,
            revisit_novelty_bonus=2.0,
            position_key_fn=mock_state_key_fn,
        )
        env.reset()

        # First visit to (5,6)
        env.step(0)
        env.step(1)  # back to (5,5)

        # Change internal state (simulating getting a new item)
        env.unwrapped.set_state_value([1.0, 0, 0, 0, 0, 0, 0, 0])

        # Revisit (5,6) with new state
        _, reward, _, _, info = env.step(0)

        assert info["is_novel_state"] is True
        assert info["is_novel_revisit"] is True
        # Should get both novelty_bonus + revisit_novelty_bonus
        assert info["novelty_bonus"] == pytest.approx(3.0, abs=0.01)

    def test_position_tracking(self):
        """Position set should grow as new positions are visited."""
        env = self._make_wrapped(position_key_fn=mock_state_key_fn)
        env.reset()

        env.step(0)  # (5,6)
        env.step(0)  # (5,7)
        _, _, _, _, info = env.step(0)  # (5,8)

        assert info["unique_positions_this_episode"] == 3

    def test_reset_clears_episode_state(self):
        env = self._make_wrapped(position_key_fn=mock_state_key_fn)
        env.reset()

        env.step(0)
        env.step(0)
        env.step(0)

        env.reset()
        # After reset, first step should be novel again
        _, _, _, _, info = env.step(0)
        assert info["is_novel_state"] is True
        assert info["unique_states_this_episode"] == 1

    def test_max_episode_states_eviction(self):
        env = self._make_wrapped(max_episode_states=3)
        env.reset()

        for _ in range(6):
            env.step(0)

        assert len(env._episode_state_hashes) <= 3

    def test_extrinsic_reward_preserved(self):
        env = self._make_wrapped(novelty_bonus=1.0)
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["extrinsic_reward"] == 0.0
        assert reward == info["novelty_bonus"]

    def test_observation_unchanged(self):
        base_env = MockGridEnv()
        wrapped_env = EpisodicStateNoveltyWrapper(
            MockGridEnv(), state_hash_fn=mock_state_hash_fn
        )
        base_obs, _ = base_env.reset()
        wrapped_obs, _ = wrapped_env.reset()

        for key in base_obs:
            if isinstance(base_obs[key], np.ndarray):
                np.testing.assert_array_equal(base_obs[key], wrapped_obs[key])
            else:
                assert base_obs[key] == wrapped_obs[key]

    def test_get_episode_stats(self):
        env = self._make_wrapped(position_key_fn=mock_state_key_fn)
        env.reset()
        env.step(0)
        stats = env.get_episode_stats()
        assert "unique_states" in stats
        assert "unique_positions" in stats
        assert "novel_revisits" in stats


# =========================================================================
# Composability tests (stacking both wrappers)
# =========================================================================

class TestComposability:
    def test_stack_both_wrappers(self):
        """Both wrappers should work together without errors."""
        env = MockGridEnv()
        env = EpisodicStateNoveltyWrapper(
            env,
            state_hash_fn=mock_state_hash_fn,
            position_key_fn=mock_state_key_fn,
            novelty_bonus=0.5,
        )
        env = CountBasedExplorationWrapper(
            env,
            state_key_fn=mock_state_key_fn,
            scale=1.0,
        )
        obs, _ = env.reset()
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            assert reward is not None
            # Should have both wrapper metrics in info
            assert "exploration_bonus" in info
            assert "novelty_bonus" in info

    def test_stacked_rewards_combine(self):
        """Total reward should include bonuses from both wrappers."""
        env = MockGridEnv()
        env = EpisodicStateNoveltyWrapper(
            env,
            state_hash_fn=mock_state_hash_fn,
            novelty_bonus=1.0,
        )
        env = CountBasedExplorationWrapper(
            env,
            state_key_fn=mock_state_key_fn,
            scale=2.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        # First visit: novelty gives 1.0, count gives 2.0/(1+1^0.5) = 1.0
        # But the count wrapper sees the novelty-augmented reward as "extrinsic"
        # The important thing is both bonuses are present
        assert info["exploration_bonus"] > 0
        assert info["novelty_bonus"] > 0
        assert reward > 0

    def test_stacked_observations_unchanged(self):
        """Observations should pass through both wrappers unchanged."""
        base_env = MockGridEnv()
        stacked = CountBasedExplorationWrapper(
            EpisodicStateNoveltyWrapper(
                MockGridEnv(),
                state_hash_fn=mock_state_hash_fn,
            ),
            state_key_fn=mock_state_key_fn,
        )
        base_obs, _ = base_env.reset()
        stacked_obs, _ = stacked.reset()

        for key in base_obs:
            if isinstance(base_obs[key], np.ndarray):
                np.testing.assert_array_equal(base_obs[key], stacked_obs[key])
            else:
                assert base_obs[key] == stacked_obs[key]


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_count_wrapper_zero_scale(self):
        """Scale=0 should add no bonus."""
        env = CountBasedExplorationWrapper(
            MockGridEnv(), state_key_fn=mock_state_key_fn, scale=0.0
        )
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0

    def test_novelty_wrapper_zero_bonus(self):
        """Bonus=0 should add no reward."""
        env = EpisodicStateNoveltyWrapper(
            MockGridEnv(),
            state_hash_fn=mock_state_hash_fn,
            novelty_bonus=0.0,
            revisit_novelty_bonus=0.0,
        )
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0

    def test_many_steps_no_crash(self):
        """Run many steps to verify no memory issues or crashes."""
        env = CountBasedExplorationWrapper(
            EpisodicStateNoveltyWrapper(
                MockGridEnv(),
                state_hash_fn=mock_state_hash_fn,
                position_key_fn=mock_state_key_fn,
            ),
            state_key_fn=mock_state_key_fn,
            scale=1.0,
        )
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
