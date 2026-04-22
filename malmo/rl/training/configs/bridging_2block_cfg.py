import os
from training.configs.bridging_cfg import BridgingCFG


class Bridging2BlockCFG(BridgingCFG):
    MISSION_FILE = os.path.join(
        BridgingCFG.ROOT_DIR, "envs", "bridging_2block", "missions", "bridging_2block.xml"
    )
    GOAL_POS      = (0.5, 46.0, 4.5)
    Z_SUCCESS     = 4.0
    Z_SUCCESS_MAX = 6.0
    BRIDGE_Z_END  = 3      # gap is Z=2..3
    MAX_STEPS     = 150
