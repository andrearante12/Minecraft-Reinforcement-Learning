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
    MAX_STEPS        = 1200
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

    # ── Actions (12 discrete) ────────────────────────────────────────────────
    BRIDGING_ACTIONS = [
        # Movement
        ("move_forward",  ["move 1"],    ["move 0"]),    # 0
        ("move_backward", ["move -1"],   ["move 0"]),    # 1
        ("strafe_left",   ["strafe -1"], ["strafe 0"]),  # 2
        ("strafe_right",  ["strafe 1"],  ["strafe 0"]),  # 3
        # Camera
        ("look_down",     ["pitch 1"],   ["pitch 0"]),   # 4
        ("look_up",       ["pitch -1"],  ["pitch 0"]),   # 5
        ("turn_left",     ["turn -1"],   ["turn 0"]),    # 6
        ("turn_right",    ["turn 1"],    ["turn 0"]),    # 7
        # Sneak/crouch — explicit press and release (ContinuousMovementCommands uses "crouch")
        ("sneak_down",    ["crouch 1"],  []),             # 8
        ("sneak_up",      ["crouch 0"],  []),             # 9
        # Block placement
        ("place_block",   ["use 1"],     ["use 0"]),     # 10
        # Idle
        ("no_op",         [],            []),             # 11
    ]
    ACTIONS   = BRIDGING_ACTIONS
    N_ACTIONS = len(ACTIONS)  # 12

    # ── Bridging-specific rewards ─────────────────────────────────────────────
    # Block placement — distinguished by position relative to the gap zone.
    # Agent z-coordinate is used as a proxy for placement location because
    # Malmo's observation lag makes the placed-block voxel unreliable on the
    # same step as placement.
    REWARD_BLOCK_PLACED_VALID   = +5.0   # block placed while in (or just before) the gap zone
    REWARD_BLOCK_PLACED_WASTED  =  0.0   # no penalty for off-target placement — any placement is exploration
    REWARD_SNEAK_PLACE          = +0.3   # bonus on valid placement when crouched (+1.3 total)

    REWARD_STEP_PENALTY  = -0.01   # reduced from -0.02: setup actions (turn-around) are less penalised
    REWARD_PROGRESS_COEF = +20.0   # per new Z-block reached — primary signal, must dominate placement

    # ── Behavioural shaping ───────────────────────────────────────────────────
    REWARD_ENTERED_GAP     = +5.0    # one-time: strong signal to leave the platform
    REWARD_SNEAK_IN_GAP    = +0.005  # per step: crouching while inside the gap
    REWARD_SNEAK_AT_EDGE   = +0.01   # per step: crouching on the platform block immediately before the gap

    # Crosshair alignment: fires once (negatively) when the agent looks away
    # from a goal-facing block face without having placed a block first.
    REWARD_ALIGNMENT_BREAK =  0.0   # removed: was suppressing camera exploration near blocks

    # Camera shaping: small per-step rewards while in the gap / at the edge.
    # sin-shaped look-down peaks at 45° pitch, preventing straight-down collapse.
    # look-back factor is 0 when facing the goal, 1 when facing fully backward.
    REWARD_LOOK_DOWN = +0.004   # max reward at 45° downward pitch
    REWARD_LOOK_BACK = +0.002   # max reward when facing directly backward (-Z)

    # Stall: additional per-step penalty when z-progress stalls beyond threshold
    STALL_THRESHOLD = 15     # steps without z-progress before penalty begins
    REWARD_STALL    = -0.02  # extra per-step cost during stall (stacks with step penalty)

    # ── Bridge geometry (gap zone) ────────────────────────────────────────────
    BRIDGE_Z_START = 2     # first Z of the gap
    BRIDGE_Z_END   = 6     # last Z of the gap (inclusive)
    BRIDGE_Y       = 45    # Y level where bridge blocks should go
    BRIDGE_X_MIN   = 0     # X range for valid bridge placement (single block wide)
    BRIDGE_X_MAX   = 0

    # ── Hyperparameter overrides for bridging ─────────────────────────────────
    ENTROPY_COEF     = 0.2     # high entropy to keep exploring placement
    N_STEPS          = 256     # short rollouts so each placement is ~33% of update data
    N_EPOCHS         = 8       # more gradient passes per rollout
    SAVE_EVERY       = 7       # ~50 episodes at ~85 steps/ep (7 * 600 = 4200 timesteps)
    TOTAL_EPISODES   = 10000   # harder task needs more episodes
