"""
training/configs/bridging_cfg.py
---------------------------------
Environment-specific config for the bridging task.
The agent must place blocks to bridge across a 5-block gap.
Inherits all shared hyperparameters from BaseCFG.
"""

import os
from training.configs.base_cfg import BaseCFG


class BridgingCFG(BaseCFG):
    # ── Mission ───────────────────────────────────────────────────────────────
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "bridging", "missions", "bridging.xml"
    )
    MALMO_PORT = 10000

    # ── Agent spawn and goal ──────────────────────────────────────────────────
    SPAWN    = (0.5, 46.0, 1.5)
    GOAL_POS = (0.5, 46.0, 7.5)

    # ── Episode termination ───────────────────────────────────────────────────
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 7.0
    Z_SUCCESS_MAX    = 9.0
    LANDING_TICKS    = 0      # instant success on reaching end platform
    MAX_STEPS        = 300
    STEP_DURATION    = 0.15

    # ── Voxel grid: x[-2:+2]=5, y[-2:+2]=5, z[-2:+5]=8 ─────────────────────
    GRID_X    = 5
    GRID_Y    = 5
    GRID_Z    = 8
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 200

    BLOCK_ENCODING = {
        "air":   0,
        "stone": 1,
    }
    GOAL_BLOCK = "stone"

    # ── Observation ──────────────────────────────────────────────────────────
    # Proprioception: 9 base (onGround, yaw, pitch, delta_y/x/z, x/y/z relative to spawn)
    #               + 1 inventory count (normalized)
    #               + 1 ray hit flag
    #               + 3 ray hit relative position
    #               + 3 ray face encoding (face_v, face_dx, face_dz)
    PROPRIOCEPTION_SIZE = 17
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE  # 220

    # ── Actions (14 discrete) ────────────────────────────────────────────────
    BRIDGING_ACTIONS = [
        # Movement
        ("move_forward",      ["move 1"],              ["move 0"]),              # 0
        ("move_backward",     ["move -1"],             ["move 0"]),              # 1
        ("strafe_left",       ["strafe -1"],           ["strafe 0"]),            # 2
        ("strafe_right",      ["strafe 1"],            ["strafe 0"]),            # 3
        # Camera
        ("look_down",         ["pitch 1"],             ["pitch 0"]),             # 4
        ("look_up",           ["pitch -1"],            ["pitch 0"]),             # 5
        ("turn_left",         ["turn -1"],             ["turn 0"]),              # 6
        ("turn_right",        ["turn 1"],              ["turn 0"]),              # 7
        # Sneak/crouch (ContinuousMovementCommands uses "crouch", not "sneak")
        ("sneak",             ["crouch 1"],             ["crouch 0"]),             # 8
        # Block placement
        ("place_block",       ["use 1"],                ["use 0"]),               # 9
        # Combined actions (self-contained: crouch included in on/off commands)
        ("sneak_forward",     ["crouch 1", "move 1"],   ["crouch 0", "move 0"]),  # 10
        ("sneak_backward",    ["crouch 1", "move -1"],  ["crouch 0", "move 0"]),  # 11
        ("sneak_place",       ["crouch 1", "use 1"],    ["crouch 0", "use 0"]),   # 12
        # Idle
        ("no_op",             [],                      []),                      # 13
    ]
    ACTIONS   = BRIDGING_ACTIONS
    N_ACTIONS = len(ACTIONS)  # 14

    # ── Bridging-specific rewards ─────────────────────────────────────────────
    REWARD_BLOCK_PLACED   = +0.5   # any block placed (unconditional; Malmo obs lag makes spatial check unreliable)
    REWARD_WASTEFUL_PLACE = -1.0   # reserved for future use
    REWARD_STEP_PENALTY   = -0.02  # slightly higher than parkour
    REWARD_PROGRESS_COEF  = +0.5   # per new z-level reached

    # ── Bridge geometry (gap zone) ────────────────────────────────────────────
    BRIDGE_Z_START = 2     # first Z of the gap
    BRIDGE_Z_END   = 6     # last Z of the gap (inclusive)
    BRIDGE_Y       = 45    # Y level where bridge blocks should go
    BRIDGE_X_MIN   = 0     # X range for valid bridge placement (single block wide)
    BRIDGE_X_MAX   = 0

    # ── Hyperparameter overrides for bridging ─────────────────────────────────
    ENTROPY_COEF     = 0.1     # more exploration for sequential dependencies
    N_STEPS          = 1024    # longer rollouts for longer episodes
    TOTAL_EPISODES   = 10000   # harder task needs more episodes
