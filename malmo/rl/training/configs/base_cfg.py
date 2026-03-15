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

    # ── Training loop ─────────────────────────────────────────────────────────
    TOTAL_EPISODES = 2000
    SAVE_EVERY     = 100
    LOG_EVERY      = 10
    EVAL_EVERY     = 200
    EVAL_EPISODES  = 20

    # ── Rewards ───────────────────────────────────────────────────────────────
    REWARD_FELL          = -5.0
    REWARD_SUCCESS       = +10.0
    REWARD_STEP_PENALTY  = -0.01
    REWARD_PROGRESS_COEF = +0.1
