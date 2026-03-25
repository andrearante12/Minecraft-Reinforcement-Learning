"""
training/configs/diagonal_medium_cfg.py
---------------------------------------
2-block gap with 1-block X offset. Harder diagonal jump.
"""

import os
from training.configs.base_cfg import BaseCFG


class DiagonalMediumCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "diagonal_medium", "missions", "diagonal_medium.xml"
    )
    MALMO_PORT       = 10000
    SPAWN            = (0.5, 46.0, 3.5)
    GOAL_POS         = (1.5, 45.0, 6.5)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 6.0
    Z_SUCCESS_MAX    = 7.0
    LANDING_TICKS    = 5
    MAX_STEPS        = 30
    STEP_DURATION    = 0.15

    # Diagonal success bounds — agent must land on the X=1 platform
    X_SUCCESS_MIN = 0.8
    X_SUCCESS_MAX = 2.2

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
