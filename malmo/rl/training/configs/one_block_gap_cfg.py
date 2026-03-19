from training.configs.base_cfg import BaseCFG
import os

class OneBlockGapCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "one_block_gap", "missions", "one_block_gap.xml"
    )
    MALMO_PORT       = 10000
    SPAWN            = (0.5, 46.0, 0.5)
    GOAL_POS         = (0.5, 45.0, 6.0)
    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 5.5
    MAX_STEPS        = 100
    STEP_DURATION    = 0.15

    GRID_X    = 5
    GRID_Y    = 4
    GRID_Z    = 6
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z

    BLOCK_ENCODING = {"air": 0, "stone": 1}
    GOAL_BLOCK     = "stone"

    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE

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