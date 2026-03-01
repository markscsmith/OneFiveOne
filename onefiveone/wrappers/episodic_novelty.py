"""
Episodic State Novelty Wrapper

A generic gymnasium wrapper that provides intrinsic reward for encountering truly
novel states within an episode. Unlike pure count-based methods, this wrapper
distinguishes between positional revisits and genuine state novelty.

The key insight: being at the same map tile with a new HM, a new party member, or
a new inventory item is a DIFFERENT state that deserves exploration credit. Standard
tile-counting misses this entirely and penalizes meaningful backtracking.

This wrapper uses two state representations:

    1. position_key_fn(obs) → hashable position
       Identifies WHERE the agent is (e.g., map + coordinates).

    2. state_hash_fn(obs) → hashable full state
       Identifies the COMPLETE state (position + inventory + flags + party, etc.)

The bonus is awarded when:
    - The full state hash is new (never seen this episode), AND optionally
    - The position has been visited before (making this a "novel revisit")

This specifically rewards backtracking with new capabilities — exactly the behavior
that standard RL struggles with.

Compatible with:
    - stable_baselines3 (PPO, DQN, QRDQN, A2C, SAC, etc.)
    - gymnasium Dict / Box / Discrete observation spaces
    - DummyVecEnv and SubprocVecEnv (each subprocess gets its own instance)
"""

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Set

import gymnasium as gym
import numpy as np


class EpisodicStateNoveltyWrapper(gym.Wrapper):
    """
    Wraps a gymnasium environment to add episodic state novelty bonuses.

    Tracks which (position, state) combinations have been seen within the current
    episode and rewards novel combinations.

    Args:
        env: The base gymnasium environment to wrap.
        state_hash_fn: A callable that takes an observation and returns a hashable
            representation of the FULL game state relevant to novelty detection.
            Should capture position AND internal state (inventory, flags, abilities).

            Examples:
                # Simple: position + first N memory bytes
                lambda obs: (obs['map_id'], int(obs['coords'][0]*32),
                             int(obs['coords'][1]*32), tuple(obs['m'][0,:20]))

                # Atari: hash of downsampled frame + RAM
                lambda obs: hash((obs['screen'].tobytes(), obs['ram'].tobytes()))

        position_key_fn: Optional callable that extracts just the positional component
            from an observation. When provided, the wrapper can distinguish between
            "novel position" (first time here) and "novel revisit" (been here before
            but with different state). If None, all novel states get the same bonus.

            Examples:
                lambda obs: (obs['map_id'], int(obs['coords'][0]*16),
                             int(obs['coords'][1]*16))

        novelty_bonus: Reward bonus for encountering a novel full state.
            Default: 0.5.
        revisit_novelty_bonus: Additional bonus specifically for novel revisits —
            i.e., returning to a known position with new internal state. Stacks with
            novelty_bonus. Only active when position_key_fn is provided.
            Default: 0.5.
        max_episode_states: Optional cap on tracked state hashes per episode.
            Prevents memory issues in very long episodes. When exceeded, oldest
            entries are dropped. None means no limit. Default: None.
        enable_logging: If True, adds novelty metrics to the info dict.
            Default: True.
    """

    def __init__(
        self,
        env: gym.Env,
        state_hash_fn: Callable[[Any], Any],
        position_key_fn: Optional[Callable[[Any], Any]] = None,
        novelty_bonus: float = 0.5,
        revisit_novelty_bonus: float = 0.5,
        max_episode_states: Optional[int] = None,
        enable_logging: bool = True,
    ):
        super().__init__(env)
        self.state_hash_fn = state_hash_fn
        self.position_key_fn = position_key_fn
        self.novelty_bonus = novelty_bonus
        self.revisit_novelty_bonus = revisit_novelty_bonus
        self.max_episode_states = max_episode_states
        self.enable_logging = enable_logging

        # Per-episode tracking
        self._episode_state_hashes: Set[Any] = set()
        self._episode_positions: Set[Any] = set()

        # Cumulative stats
        self._total_novelty_bonus = 0.0
        self._total_revisit_bonus = 0.0
        self._novel_state_count = 0
        self._novel_revisit_count = 0

        # For ordered eviction when max_episode_states is set
        self._state_order: list = []

    def render(self, *args, **kwargs):
        """Pass through to base env render, forwarding any extra arguments."""
        return self.env.render(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Hash the full state
        state_hash = self.state_hash_fn(obs)

        bonus = 0.0
        is_novel_state = state_hash not in self._episode_state_hashes
        is_novel_revisit = False

        if is_novel_state:
            # This full state has never been seen this episode
            self._episode_state_hashes.add(state_hash)
            self._state_order.append(state_hash)
            self._novel_state_count += 1
            bonus += self.novelty_bonus

            # Check if this is a revisit of a known position with novel state
            if self.position_key_fn is not None:
                position_key = self.position_key_fn(obs)
                if position_key in self._episode_positions:
                    # We've been to this POSITION before, but the full STATE is new.
                    # This is the backtracking-with-progress signal.
                    is_novel_revisit = True
                    bonus += self.revisit_novelty_bonus
                    self._novel_revisit_count += 1
                else:
                    self._episode_positions.add(position_key)

            # Evict oldest if over capacity
            if (
                self.max_episode_states is not None
                and len(self._episode_state_hashes) > self.max_episode_states
            ):
                self._evict_oldest()

        total_reward = reward + bonus
        self._total_novelty_bonus += bonus

        if self.enable_logging:
            info["novelty_bonus"] = bonus
            info["is_novel_state"] = is_novel_state
            info["is_novel_revisit"] = is_novel_revisit
            info["unique_states_this_episode"] = len(self._episode_state_hashes)
            info["unique_positions_this_episode"] = len(self._episode_positions)
            info["novel_revisit_count"] = self._novel_revisit_count
            info["total_novelty_bonus"] = self._total_novelty_bonus
            info["extrinsic_reward"] = reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Clear all per-episode tracking
        self._episode_state_hashes.clear()
        self._episode_positions.clear()
        self._state_order.clear()
        self._total_novelty_bonus = 0.0
        self._total_revisit_bonus = 0.0
        self._novel_state_count = 0
        self._novel_revisit_count = 0

        return self.env.reset(**kwargs)

    def _evict_oldest(self):
        """Remove the oldest state hashes to stay under max_episode_states."""
        while (
            self.max_episode_states is not None
            and len(self._episode_state_hashes) > self.max_episode_states
            and self._state_order
        ):
            oldest = self._state_order.pop(0)
            self._episode_state_hashes.discard(oldest)

    def get_episode_stats(self) -> Dict[str, Any]:
        """Return summary statistics about novelty detection this episode."""
        return {
            "unique_states": len(self._episode_state_hashes),
            "unique_positions": len(self._episode_positions),
            "novel_states_found": self._novel_state_count,
            "novel_revisits": self._novel_revisit_count,
            "total_novelty_bonus": self._total_novelty_bonus,
        }
