"""
Preset state key and hash functions for common environments.

These are ready-to-use callables for the CountBasedExplorationWrapper and
EpisodicStateNoveltyWrapper. Each function takes an observation dict and returns
a hashable value suitable for counting or novelty detection.

For custom environments, use these as templates to write your own extractors.

Usage:
    from wrappers.presets import pokemon_state_key_fn, pokemon_state_hash_fn

    env = CountBasedExplorationWrapper(env, state_key_fn=pokemon_state_key_fn)
    env = EpisodicStateNoveltyWrapper(env, state_hash_fn=pokemon_state_hash_fn,
                                       position_key_fn=pokemon_state_key_fn)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pokemon Red/Blue/Yellow (PyBoyEnv)
# ---------------------------------------------------------------------------
# PyBoyEnv observation space:
#   "m":              (16, 5608) float32  — 16 frames of game memory (normalized)
#   "s":              (36, 40, 4) float32  — game screen (normalized)
#   "map_id":         Discrete(256)        — current map ID
#   "coords":         Box(2,) float32      — player X,Y normalized [0,1]
#   "map_one_hot":    Box(256,) float32    — one-hot of current map
#   "visited_map":    Box(9,9) float32     — local reward heatmap
#   "text_onscreen":  Box(1,) float32      — dialog active?
#   "is_in_battle":   Box(1,) float32      — in combat?
# ---------------------------------------------------------------------------


def pokemon_state_key_fn(obs) -> tuple:
    """
    Extract a positional state key for count-based exploration.

    Returns (map_id, x_bucket, y_bucket) where coordinates are discretized
    to the original game tile resolution. Since coords are normalized to [0,1]
    and the original space is 256 values, we multiply back by 256.

    This means each unique game tile on each map is a distinct state for
    counting purposes.
    """
    map_id = int(obs["map_id"])
    x = int(obs["coords"][0] * 255)
    y = int(obs["coords"][1] * 255)
    return (map_id, x, y)


def pokemon_state_hash_fn(obs) -> tuple:
    """
    Create a comprehensive state hash for episodic novelty detection.

    Captures position AND game state so that revisiting a tile with new items,
    party members, or story flags registers as a genuinely novel state.

    The memory slice obs["m"][0, :64] captures the most recent frame's first
    64 bytes of the combined memory block, which includes party pokemon data.
    We quantize to 8-bit buckets to avoid floating-point noise creating false
    novelty (the values are normalized 0-1 from original 0-255 bytes).
    """
    map_id = int(obs["map_id"])
    x = int(obs["coords"][0] * 255)
    y = int(obs["coords"][1] * 255)

    # Quantize memory back to ~byte resolution and take a meaningful slice.
    # The first 264 bytes of the memory block are my_pokemon (party data),
    # followed by pokedex (38 bytes), items (42 bytes), money (4 bytes).
    # Total: ~348 bytes of highly relevant game state.
    # We sample every 4th byte to keep the hash compact but representative.
    m_slice = obs["m"][0]  # Most recent memory frame
    quantized = (m_slice[:348:4] * 255).astype(np.uint8)

    return (map_id, x, y, quantized.tobytes())


# ---------------------------------------------------------------------------
# Generic grid world presets
# ---------------------------------------------------------------------------


def grid_xy_key_fn(obs, x_key="x", y_key="y") -> tuple:
    """
    Simple (x, y) state key for grid world environments with named coordinates.

    Usage:
        env = CountBasedExplorationWrapper(
            env, state_key_fn=lambda obs: grid_xy_key_fn(obs, 'row', 'col')
        )
    """
    return (obs[x_key], obs[y_key])


def flat_obs_hash_fn(obs, resolution: int = 32) -> tuple:
    """
    Hash a flat (Box) observation by quantizing to a fixed resolution.

    Useful for continuous observation spaces where you want count-based
    exploration but don't have named state components.

    Usage:
        env = CountBasedExplorationWrapper(
            env, state_key_fn=lambda obs: flat_obs_hash_fn(obs, resolution=16)
        )
    """
    if isinstance(obs, np.ndarray):
        quantized = (obs * resolution).astype(np.int32)
        return tuple(quantized.flat)
    # If obs is a dict, flatten all values
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            parts.extend((val * resolution).astype(np.int32).flat)
        else:
            parts.append(int(val))
    return tuple(parts)
