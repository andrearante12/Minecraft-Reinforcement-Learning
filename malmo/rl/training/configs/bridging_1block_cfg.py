import os
from training.configs.bridging_cfg import BridgingCFG


class Bridging1BlockCFG(BridgingCFG):
    MISSION_FILE = os.path.join(
        BridgingCFG.ROOT_DIR, "envs", "bridging_1block", "missions", "bridging_1block.xml"
    )
    GOAL_POS      = (0.5, 46.0, 3.5)
    Z_SUCCESS     = 3.0
    Z_SUCCESS_MAX = 5.0
    BRIDGE_Z_END  = 2      # gap is only Z=2
    MAX_STEPS     = 100
