"""
training/configs/base_cfg.py
-----------------------------
Shared hyperparameters inherited by all environment configs.
Environment-specific settings (SPAWN, MISSION_FILE, etc.) are defined
in each env's own config file.
"""

import os


class BaseCFG:
    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR        = os.path.join(ROOT_DIR, "logs")
    MALMO_PYTHON   = os.path.join(ROOT_DIR, "..", "Python_Examples")

    # ── Network ───────────────────────────────────────────────────────────────
    HIDDEN_SIZE      = 128        # kept for backward compat / old model
    PROPRIO_HIDDEN   = 64         # proprio stream width
    GOAL_HIDDEN      = 64         # goal stream width
    VOXEL_HIDDEN     = 128        # voxel stream width
    HEAD_HIDDEN      = 256        # actor & critic head width

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
    ENTROPY_COEF = 0.05

    # ── PPO improvements ───────────────────────────────────────────────────
    # Normalization
    OBS_NORM         = True
    REWARD_NORM      = True
    NORM_CLIP        = 10.0

    # LR scheduling
    LR_DECAY         = True
    LR_END           = 0.0

    # Entropy scheduling
    ENTROPY_DECAY    = True
    ENTROPY_COEF_END = 0.001

    # ── DQN ───────────────────────────────────────────────────────────────────
    BUFFER_CAPACITY     = 10000
    EPSILON_START       = 1.0
    EPSILON_END         = 0.05
    EPSILON_DECAY_STEPS = 5000
    TARGET_UPDATE_FREQ  = 500

    # ── Training loop ─────────────────────────────────────────────────────────
    TOTAL_EPISODES = 5000
    SAVE_EVERY     = 100
    LOG_EVERY      = 10
    EVAL_EVERY     = 200
    EVAL_EPISODES  = 20

    # ── Rewards ───────────────────────────────────────────────────────────────
    REWARD_FELL          = -5.0
    REWARD_SUCCESS       = +10.0
    REWARD_STEP_PENALTY  = -0.01
    REWARD_PROGRESS_COEF = +0.5
    REWARD_TIMEOUT       = -5.0

    # Proximity-scaled terminal penalties: scale fell/timeout by how far
    # the agent got.  Close to goal = small penalty, at spawn = full penalty.
    PROXIMITY_SCALED_TERMINAL = True

    # Near-miss bonus: positive reward when agent times out close to the goal
    NEAR_MISS_THRESHOLD = 1.5    # blocks from Z_SUCCESS
    REWARD_NEAR_MISS    = +2.0

    # ── Environment behavior flags ───────────────────────────────────────────
    SUCCESS_REQUIRES_ON_GROUND = False   # True = check OnGround, False = check y >= FALL_Y_THRESHOLD
    REWARD_ON_MISSION_ENDED    = True    # True = assign REWARD_TIMEOUT when mission ends unexpectedly

