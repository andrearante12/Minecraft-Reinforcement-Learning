"""
training/configs/three_block_gap_cfg.py
----------------------------------------
Environment-specific config for the three-block gap task.
Identical to simple_jump except the landing platform starts
one block further (Z=7 instead of Z=6), making the gap 3 blocks wide.
"""

import os
from training.configs.base_cfg import BaseCFG


class ThreeBlockGapCFG(BaseCFG):
    # ── Mission ───────────────────────────────────────────────────────────────
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "three_block_gap", "missions", "three_block_gap.xml"
    )
    MALMO_PORT = 10000

    # ── Agent spawn and goal ──────────────────────────────────────────────────
    SPAWN    = (0.5, 46.0, 3.5)
    GOAL_POS = (0.5, 45.0, 7.5)   # landing block center

    # ── Episode termination ───────────────────────────────────────────────────
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 7.0         # landing block at Z=7
    Z_SUCCESS_MAX    = 8.0
    LANDING_TICKS    = 5
    MAX_STEPS        = 30
    STEP_DURATION    = 0.15

    # ── Voxel grid: x[-2:+2]=5, y[-1:+3]=5, z[-2:+3]=6 ──────────────────────
    GRID_X    = 5
    GRID_Y    = 5
    GRID_Z    = 6
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 150

    BLOCK_ENCODING = {
        "air":   0,
        "stone": 1,
    }
    GOAL_BLOCK = "stone"

    # Observation: proprioception(6) + goal_delta(3) + voxels(150) = 159
    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE

    # ── Actions ──────────────────────────────────────────────────────────────
    ACTIONS   = BaseCFG.DEFAULT_ACTIONS
    N_ACTIONS = len(ACTIONS)

    # ── Behavior overrides ──────────────────────────────────────────────────
    SUCCESS_REQUIRES_ON_GROUND = True
    REWARD_ON_MISSION_ENDED    = False

    # ── Three-block-gap tuning ───────────────────────────────────────────
    # Resumed training: start with lower LR/entropy since policy is already trained
    TOTAL_EPISODES   = 10000
    ENTROPY_COEF     = 0.02    # lowered from 0.05 for resume
    ENTROPY_COEF_END = 0.005   # keep a higher floor than default — 3-block still needs exploration
    LR               = 1e-4    # lowered from 3e-4 for resume
    LR_END           = 1e-5    # don't decay to zero
