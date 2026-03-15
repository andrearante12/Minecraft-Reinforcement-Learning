"""
training/config.py
------------------
Single source of truth for all hyperparameters and path constants.
"""

import os


class CFG:
    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MISSION_FILE   = os.path.join(ROOT_DIR, "simple_jump", "missions", "parkour_simple_jump.xml")
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR        = os.path.join(ROOT_DIR, "logs")
    MALMO_PYTHON   = os.path.join(ROOT_DIR, "..", "Python_Examples")

    # ── Environment ───────────────────────────────────────────────────────────
    SPAWN            = (0.5, 46.0, 0.5)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 5.0
    GOAL_BLOCK       = "stone"
    STEP_DURATION    = 0.15
    MAX_STEPS        = 200
    MALMO_PORT       = 10000

    # Voxel grid: x[-2:+2]=5, y[-1:+2]=4, z[-2:+3]=6
    GRID_X    = 5
    GRID_Y    = 4
    GRID_Z    = 6
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 120

    BLOCK_ENCODING = {
        "air":   0,
        "stone": 1,
    }

    # Observation: proprioception(6) + goal_delta(3) + voxels(120) = 129
    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE

    GOAL_POS = (0.5, 45.0, 7.0)

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

    # ── Network ───────────────────────────────────────────────────────────────
    HIDDEN_SIZE = 128

    # ── Shared hyperparameters ────────────────────────────────────────────────
    GAMMA         = 0.99
    LR            = 3e-4
    BATCH_SIZE    = 64
    MAX_GRAD_NORM = 0.5

    # ── PPO ───────────────────────────────────────────────────────────────────
    N_STEPS      = 512
    N_EPOCHS     = 4
    GAE_LAMBDA   = 0.95
    CLIP_EPS     = 0.2
    VALUE_COEF   = 0.5
    ENTROPY_COEF = 0.01

    # ── DQN ───────────────────────────────────────────────────────────────────
    BUFFER_CAPACITY     = 10000
    EPSILON_START       = 1.0
    EPSILON_END         = 0.05
    EPSILON_DECAY_STEPS = 5000
    TARGET_UPDATE_FREQ  = 500

    # ── Training ──────────────────────────────────────────────────────────────
    TOTAL_EPISODES = 2000
    SAVE_EVERY     = 100
    LOG_EVERY      = 10
    EVAL_EVERY     = 200

    # ── Rewards ───────────────────────────────────────────────────────────────
    REWARD_FELL          = -5.0
    REWARD_SUCCESS       = +10.0
    REWARD_STEP_PENALTY  = -0.01
    REWARD_PROGRESS_COEF = +0.1

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_EPISODES = 20


# Flat exports for convenient imports
INPUT_SIZE  = CFG.INPUT_SIZE
HIDDEN_SIZE = CFG.HIDDEN_SIZE
N_ACTIONS   = CFG.N_ACTIONS
ACTIONS     = CFG.ACTIONS
SPAWN       = CFG.SPAWN