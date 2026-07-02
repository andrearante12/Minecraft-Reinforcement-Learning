"""
training/configs/hunting_wild_cfg.py
--------------------------------------
Config for the hunting task on natural Minecraft terrain (hunting_wild env).

Same task as HuntingCFG (penned/wandering/fleeing pig, kill for reward) but
in a reproducible natural world instead of a superflat walled arena. The agent
must navigate real terrain (hills, trees, water) while still being bounded by
a bedrock cage to keep the episode tractable.

Research purpose: does the world-model agent transfer when the dynamics include
non-trivial terrain, and does the aleatoric/epistemic split still track pig
predictability correctly on rough ground?

Inherits everything from HuntingCFG and overrides only the world-specific bits.
If the agent spawns in the air or underground, adjust SPAWN[1] and
FALL_Y_THRESHOLD by measuring ObservationFromFullStats.YPos at first step.
"""

import os
from training.configs.hunting_cfg import HuntingCFG


class HuntingWildCFG(HuntingCFG):
    # ── Mission ───────────────────────────────────────────────────────────────
    MISSION_FILE = os.path.join(
        HuntingCFG.ROOT_DIR, "envs", "hunting_wild", "missions", "hunting_wild.xml"
    )

    # ── Agent spawn (calibrate y for seed 4837 plains) ────────────────────────
    # Run once, check YPos in logs, then set SPAWN[1] = observed_ground_y + 1.
    SPAWN           = (0.5, 65.0, 0.5)
    FALL_Y_THRESHOLD = 60.0    # terrain varies; bedrock floor at y=61

    # ── Larger play area (42×42 bedrock cage) ────────────────────────────────
    ARENA_MIN       = -18.0
    ARENA_MAX       = 18.0
    FLEEING_MIN_DIST = 8.0     # slightly larger min-distance for bigger arena

    # ── Voxel grid is unchanged (5×3×5 = 75 dims) ────────────────────────────
    # The y range [-1:+1] may miss tall obstacles on hilly terrain. Expand
    # to 7×5×7 for a follow-up study (updates INPUT_SIZE and invalidates ckpts).

    # ── Block encoding: natural terrain blocks ────────────────────────────────
    # Unknown blocks default to 1 (solid, treat as obstacle).
    BLOCK_ENCODING = {
        # passable
        "air":        0,
        "water":      0,
        "flowing_water": 0,
        # common solid ground
        "grass":      1,
        "dirt":       1,
        "stone":      1,
        "gravel":     1,
        "sand":       1,
        "sandstone":  1,
        "cobblestone": 1,
        "clay":       1,
        # tree materials (solid but climb-able)
        "log":        2,
        "leaves":     2,
        # cage wall marker
        "bedrock":    3,
    }

    # ── Notes ─────────────────────────────────────────────────────────────────
    # Reward constants inherited from HuntingCFG. Approach reward may be noisier
    # on rough terrain (true distance closes slower when navigating obstacles).
    # The aim shaping (AIM_ALIGN_COEF) is more critical here because terrain
    # can obstruct the line-of-sight and force the agent to reposition.
