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
    SPAWN    = (0.5, 46.0, 0.5)
    GOAL_POS = (0.5, 45.0, 8.0)   # landing platform center (one block further than simple_jump)

    # ── Episode termination ───────────────────────────────────────────────────
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 7.0         # landing platform starts at Z=7
    MAX_STEPS        = 200
    STEP_DURATION    = 0.15

    # ── Voxel grid: x[-2:+2]=5, y[-1:+2]=4, z[-2:+3]=6 ──────────────────────
    GRID_X    = 5
    GRID_Y    = 4
    GRID_Z    = 6
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 120

    BLOCK_ENCODING = {
        "air":   0,
        "stone": 1,
    }
    GOAL_BLOCK = "stone"

    # Observation: proprioception(6) + goal_delta(3) + voxels(120) = 129
    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE

    # ── Actions (name, commands_on, commands_off) ─────────────────────────────
    ACTIONS = [
        ("move_forward",   ["move 1"],                       ["move 0"]),
        ("move_backward",  ["move -1"],                      ["move 0"]),
        ("strafe_left",    ["strafe -1"],                    ["strafe 0"]),
        ("strafe_right",   ["strafe 1"],                     ["strafe 0"]),
        ("sprint_forward", ["sprint 1", "move 1"],           ["sprint 0", "move 0"]),
        ("jump",           ["jump 1"],                       ["jump 0"]),
        ("sprint_jump",    ["sprint 1", "move 1", "jump 1"], ["sprint 0", "move 0", "jump 0"]),
        ("look_down",      ["pitch 1"],                      ["pitch 0"]),
        ("look_up",        ["pitch -1"],                     ["pitch 0"]),
        ("turn_left",      ["turn -1"],                      ["turn 0"]),
        ("turn_right",     ["turn 1"],                       ["turn 0"]),
        ("no_op",          [],                               []),
    ]
    N_ACTIONS = len(ACTIONS)