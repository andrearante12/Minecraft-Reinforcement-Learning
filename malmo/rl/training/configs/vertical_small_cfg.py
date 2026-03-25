"""
training/configs/vertical_small_cfg.py
--------------------------------------
1-block gap with 1-block height increase. Tests upward jumping.
"""

import os
from training.configs.base_cfg import BaseCFG


class VerticalSmallCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "vertical_small", "missions", "vertical_small.xml"
    )
    MALMO_PORT       = 10000
    SPAWN            = (0.5, 46.0, 3.5)
    GOAL_POS         = (0.5, 47.0, 5.5)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 5.0
    Z_SUCCESS_MAX    = 6.0
    LANDING_TICKS    = 5
    MAX_STEPS        = 30
    STEP_DURATION    = 0.15

    # Vertical success bound — agent must be on the higher platform (Y=46+1)
    Y_SUCCESS_MIN = 46.5

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
