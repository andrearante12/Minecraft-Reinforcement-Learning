"""
training/config.py
------------------

Usage:
    from training.config import CFG
    print(CFG.LR)
    print(CFG.SPAWN)
"""

import os


class CFG:
    # Paths
    # Root of the parkour project (one level up from this file)
    ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MISSION_FILE = os.path.join(ROOT_DIR, "simple_jump", "missions", "parkour_simple_jump.xml")
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR        = os.path.join(ROOT_DIR, "logs")

    # Path to Malmo's Python_Examples so MalmoPython can be imported
    MALMO_PYTHON = os.path.join(ROOT_DIR, "..", "Python_Examples")

    # Environment 
    SPAWN             = (0.5, 46.0, 0.5)   
    FALL_Y_THRESHOLD  = 43.0               
    Z_SUCCESS         = 5.0               # Z coordinate of a success
    GOAL_BLOCK        = "stone"           # block type at the landing platform
    STEP_DURATION     = 0.15              
    MAX_STEPS         = 200              # max steps per episode before timeout
    MALMO_PORT        = 10000            # Minecraft client port

    # Observation Grid Dimensions 
    # x: -2 to +2 = 5,  y: -1 to +2 = 4,  z: -2 to +3 = 6
    GRID_X = 5   # x: -2 to +2
    GRID_Y = 4   # y: -1 to +2
    GRID_Z = 6   # z: -2 to +3
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 120

    BLOCK_ENCODING = {
        "air":        0,
        "stone":      1,
        # anything else → 1 (treat as solid)
    }

    # Observation Vector
    # Proprioception: onGround, yaw, pitch, dy, dx, dz   = 6
    # Goal delta:     goal_dx, goal_dy, goal_dz          = 3
    # Voxel grid:     GRID_SIZE                          = 120
    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE  # 129

    # Goal position — center of the landing platform in your XML
    GOAL_POS = (0.5, 45.0, 7.0)   # (x, y, z) of landing platform center

    # Actions 
    # (name, commands_on, commands_off)
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
    N_ACTIONS = len(ACTIONS)  # 12

    # Network Size
    HIDDEN_SIZE = 128   

    # PPO hyperparameters 
    N_STEPS        = 512    
    BATCH_SIZE     = 64     
    N_EPOCHS       = 4      
    GAMMA          = 0.99  
    GAE_LAMBDA     = 0.95   
    CLIP_EPS       = 0.2    
    LR             = 3e-4   
    VALUE_COEF     = 0.5   
    ENTROPY_COEF   = 0.01   
    MAX_GRAD_NORM  = 0.5    

    # Training loop
    TOTAL_EPISODES    = 2000   
    SAVE_EVERY        = 100    # save checkpoint every N episodes
    LOG_EVERY         = 10     # print summary every N episodes
    EVAL_EVERY        = 200    # run evaluation every N episodes

    # Rewards
    REWARD_FELL          = -5.0    
    REWARD_SUCCESS       = +10.0  
    REWARD_STEP_PENALTY  = -0.01   
    REWARD_PROGRESS_COEF = +0.1    

    # Evaluation
    EVAL_EPISODES = 20   # episodes to run during evaluation


# Expose flat constants for import
INPUT_SIZE  = CFG.INPUT_SIZE
HIDDEN_SIZE = CFG.HIDDEN_SIZE
N_ACTIONS   = CFG.N_ACTIONS
ACTIONS     = CFG.ACTIONS
SPAWN       = CFG.SPAWN