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

    # ── Behavioral Cloning ─────────────────────────────────────────────────────
    DEMO_PATH     = None    # path to demo JSON (required for BC)
    BC_EPOCHS     = 10      # passes over demo data per update
    BC_BATCH_SIZE = 64      # minibatch size for BC updates

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

    # ── Default action space (15 discrete) ────────────────────────────────────
    DEFAULT_ACTIONS = [
        ("move_forward",      ["move 1"],                                     ["move 0"]),
        ("move_backward",     ["move -1"],                                    ["move 0"]),
        ("strafe_left",       ["strafe -1"],                                  ["strafe 0"]),
        ("strafe_right",      ["strafe 1"],                                   ["strafe 0"]),
        ("sprint_forward",    ["sprint 1", "move 1"],                         ["sprint 0", "move 0"]),
        ("jump",              ["jump 1"],                                     ["jump 0"]),
        ("sprint_jump",       ["sprint 1", "move 1", "jump 1"],              ["sprint 0", "move 0", "jump 0"]),
        ("jump_forward",      ["move 1", "jump 1"],                          ["move 0", "jump 0"]),
        ("sprint_jump_left",  ["sprint 1", "move 1", "strafe -1", "jump 1"], ["sprint 0", "move 0", "strafe 0", "jump 0"]),
        ("sprint_jump_right", ["sprint 1", "move 1", "strafe 1", "jump 1"],  ["sprint 0", "move 0", "strafe 0", "jump 0"]),
        ("look_down",         ["pitch 1"],                                    ["pitch 0"]),
        ("look_up",           ["pitch -1"],                                   ["pitch 0"]),
        ("turn_left",         ["turn -1"],                                    ["turn 0"]),
        ("turn_right",        ["turn 1"],                                     ["turn 0"]),
        ("no_op",             [],                                             []),
    ]

    # ── Default voxel grid dimensions ──────────────────────────────────────────
    DEFAULT_GRID_Y = 5   # Y[-1:+3] = 5 blocks vertical visibility

    # ── Environment behavior flags ───────────────────────────────────────────
    SUCCESS_REQUIRES_ON_GROUND = False   # True = check OnGround, False = check y >= FALL_Y_THRESHOLD
    REWARD_ON_MISSION_ENDED    = True    # True = assign REWARD_TIMEOUT when mission ends unexpectedly

    # ── Optional success bounds (for diagonal / vertical envs) ───────────────
    X_SUCCESS_MIN = None   # if set, success requires x >= this value
    X_SUCCESS_MAX = None   # if set, success requires x <= this value
    Y_SUCCESS_MIN = None   # if set, success requires y >= this value (overrides FALL_Y_THRESHOLD for success)

    # ── Landing phase ──────────────────────────────────────────────────────────
    LANDING_TICKS       = 0      # 0 = instant success (backward compat), >0 = must stay on block
    REWARD_LANDING_TICK = +0.5   # per-tick reward for staying on the landing block
    Z_SUCCESS_MAX       = None   # upper Z bound for landing zone (None = no upper bound)

