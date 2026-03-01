"""
Count-Based Exploration Bonus Wrapper

A generic gymnasium wrapper that adds intrinsic reward based on state visitation
counts. Encourages revisiting under-explored states and makes backtracking through
known territory worthwhile by providing a decaying bonus.

This wrapper is environment-agnostic. The caller provides a `state_key_fn` that
extracts a hashable state representation from the observation. This allows the
same wrapper to work with any gymnasium environment — grid worlds, Atari, Pokemon,
or custom envs.

The bonus formula is:

    bonus = scale / (1 + visit_count ** exponent)

where `exponent` controls how aggressively the bonus decays with repeated visits.
With the default exponent of 0.5 (square root), the bonus follows:

    visit 1 → scale / 2.0
    visit 4 → scale / 3.0
    visit 9 → scale / 4.0

This ensures that even well-trodden paths retain *some* reward, which is the key
insight for solving the backtracking problem: the agent can traverse known territory
to reach new frontiers without facing a reward desert.

Compatible with:
    - stable_baselines3 (PPO, DQN, QRDQN, A2C, SAC, etc.)
    - gymnasium Dict / Box / Discrete observation spaces
    - DummyVecEnv and SubprocVecEnv (each subprocess gets its own instance)
"""

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CountBasedExplorationWrapper(gym.Wrapper):
    """
    Wraps a gymnasium environment to add count-based intrinsic exploration bonuses.

    Args:
        env: The base gymnasium environment to wrap.
        state_key_fn: A callable that takes an observation and returns a hashable
            key representing the agent's "position" or abstract state for counting
            purposes. This is what gets counted.

            Examples:
                # For a grid world with (x, y) in the obs dict:
                lambda obs: (obs['x'], obs['y'])

                # For Atari with pixel observations (hash downsampled frame):
                lambda obs: hash(obs.tobytes())

                # For Pokemon with map + coords:
                lambda obs: (obs['map_id'], int(obs['coords'][0]*16), int(obs['coords'][1]*16))

        scale: Multiplier for the exploration bonus. Larger values make exploration
            more attractive relative to extrinsic rewards. Default: 1.0.
        exponent: Controls decay speed. Lower values = slower decay = more persistent
            bonus. Default: 0.5 (square root decay).
        decay_on_reset: Factor applied to all visit counts when the environment resets.
            1.0 means counts persist fully across episodes. 0.0 means counts reset
            completely. 0.95 means 5% forgetting per episode. Default: 1.0.
        max_states: Optional cap on the number of tracked states. When exceeded, the
            least-visited states are pruned. Prevents unbounded memory growth in very
            long training runs. None means no limit. Default: None.
        enable_logging: If True, adds exploration metrics to the info dict returned
            by step(). Default: True.
    """

    def __init__(
        self,
        env: gym.Env,
        state_key_fn: Callable[[Any], Any],
        scale: float = 1.0,
        exponent: float = 0.5,
        decay_on_reset: float = 1.0,
        max_states: Optional[int] = None,
        enable_logging: bool = True,
    ):
        super().__init__(env)
        self.state_key_fn = state_key_fn
        self.scale = scale
        self.exponent = exponent
        self.decay_on_reset = decay_on_reset
        self.max_states = max_states
        self.enable_logging = enable_logging

        # State tracking
        self.visit_counts: Dict[Any, float] = defaultdict(float)
        self.total_bonus_awarded = 0.0
        self.steps_since_reset = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract the abstract state key from the observation
        state_key = self.state_key_fn(obs)

        # Increment visit count
        self.visit_counts[state_key] += 1.0
        count = self.visit_counts[state_key]

        # Calculate intrinsic bonus: decays with visit count
        bonus = self.scale / (1.0 + count ** self.exponent)

        # Combine with extrinsic reward
        total_reward = reward + bonus

        self.total_bonus_awarded += bonus
        self.steps_since_reset += 1

        # Prune if we've exceeded the state cap
        if self.max_states is not None and len(self.visit_counts) > self.max_states:
            self._prune_least_visited()

        # Add metrics to info dict for logging/tensorboard
        if self.enable_logging:
            info["exploration_bonus"] = bonus
            info["state_visit_count"] = count
            info["unique_states_visited"] = len(self.visit_counts)
            info["total_exploration_bonus"] = self.total_bonus_awarded
            info["extrinsic_reward"] = reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Apply decay to visit counts between episodes
        if self.decay_on_reset < 1.0:
            pruned = {}
            for key, count in self.visit_counts.items():
                decayed = count * self.decay_on_reset
                if decayed >= 0.5:  # Drop states that have decayed to near-zero
                    pruned[key] = decayed
            self.visit_counts = defaultdict(float, pruned)

        self.total_bonus_awarded = 0.0
        self.steps_since_reset = 0

        return self.env.reset(**kwargs)

    def _prune_least_visited(self):
        """Remove the least-visited states to stay under max_states."""
        if self.max_states is None:
            return
        while len(self.visit_counts) > self.max_states:
            # Find and remove the state with the lowest visit count
            min_key = min(self.visit_counts, key=self.visit_counts.get)
            del self.visit_counts[min_key]

    def get_visit_counts(self) -> Dict[Any, float]:
        """Return a copy of the current visit counts. Useful for debugging."""
        return dict(self.visit_counts)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about exploration state."""
        counts = list(self.visit_counts.values())
        if not counts:
            return {
                "unique_states": 0,
                "total_visits": 0,
                "mean_visits": 0.0,
                "max_visits": 0,
                "total_bonus": self.total_bonus_awarded,
            }
        return {
            "unique_states": len(counts),
            "total_visits": sum(counts),
            "mean_visits": float(np.mean(counts)),
            "max_visits": max(counts),
            "total_bonus": self.total_bonus_awarded,
        }
