from training.configs.base_cfg import BaseCFG
import os

class OneBlockGapCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "one_block_gap", "missions", "one_block_gap.xml"
    )
    MALMO_PORT       = 10000
    SPAWN            = (0.5, 46.0, 3.5)
    GOAL_POS         = (0.5, 45.0, 5.5)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 5.0
    Z_SUCCESS_MAX    = 6.0
    LANDING_TICKS    = 5
    MAX_STEPS        = 30
    STEP_DURATION    = 0.15

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