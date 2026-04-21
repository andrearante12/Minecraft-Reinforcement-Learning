"""
training/configs/multi_jump_course_cfg.py
-----------------------------------------
Multi-jump evaluation course: 4 chained jumps (1-block, 2-block, diagonal, vertical).
Single-block platforms matching training geometry.
"""

import os
from training.configs.base_cfg import BaseCFG


class MultiJumpCourseCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "multi_jump_course", "missions", "multi_jump_course.xml"
    )
    MALMO_PORT       = 10000
    SPAWN            = (0.5, 46.0, 3.5)
    GOAL_POS         = (1.5, 46.0, 12.5)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 12.0
    Z_SUCCESS_MAX    = 13.0
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

    ACTIONS          = BaseCFG.DEFAULT_ACTIONS
    N_ACTIONS        = len(ACTIONS)

    # Z thresholds where the agent is considered to have landed each jump
    JUMP_CHECKPOINTS = [5.0, 8.0, 10.0, 12.0]
