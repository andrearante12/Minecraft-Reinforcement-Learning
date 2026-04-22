"""
training/configs/multi_jump_branch_cfg.py
-----------------------------------------
Three asymmetric branches from a shared start, all converging on one goal block.
Adjacent branches are within diagonal-jump range at multiple points, letting the
agent discover and switch between paths mid-run.

Branch difficulty:
  Left  (x=-1): easiest  — diagonal start, 4x 1-block gaps, diagonal finish
  Center(x= 0): medium   — straight start, 1-block then 2-block gaps
  Right (x= 1): hardest  — diagonal start, 3x 2-block gaps, diagonal finish
"""

import os
from training.configs.base_cfg import BaseCFG


class MultiJumpBranchCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "multi_jump_branch", "missions", "multi_jump_branch.xml"
    )
    MALMO_PORT       = 10000
    SPAWN            = (0.5, 46.0, 3.5)
    GOAL_POS         = (0.5, 45.0, 13.5)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 13.0
    Z_SUCCESS_MAX    = 14.0
    MAX_STEPS        = 120
    STEP_DURATION    = 0.15
    LANDING_TICKS    = 5

    GRID_X    = 5
    GRID_Y    = 5
    GRID_Z    = 6
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 150

    BLOCK_ENCODING = {"air": 0, "stone": 1}
    GOAL_BLOCK     = "stone"

    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE

    ACTIONS   = BaseCFG.DEFAULT_ACTIONS
    N_ACTIONS = len(ACTIONS)

    # Z thresholds that mark progress regardless of which branch is taken
    JUMP_CHECKPOINTS = [5.0, 8.0, 11.0, 13.0]
