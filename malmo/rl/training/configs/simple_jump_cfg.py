"""
training/configs/simple_jump_cfg.py
------------------------------------
Environment-specific config for the simple jump task.
Inherits all shared hyperparameters from BaseCFG.
Only override what is different for this environment.
"""

import os
from training.configs.base_cfg import BaseCFG


class SimpleJumpCFG(BaseCFG):
    # ── Mission ───────────────────────────────────────────────────────────────
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "simple_jump", "missions", "parkour_simple_jump.xml"
    )
    MALMO_PORT = 10000

    # ── Agent spawn and goal ──────────────────────────────────────────────────
    SPAWN    = (0.5, 46.0, 3.5)
    GOAL_POS = (0.5, 45.0, 6.5)

    # ── Episode termination ───────────────────────────────────────────────────
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 6.0
    Z_SUCCESS_MAX    = 7.0
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
