import os
from training.configs.bridging_cfg import BridgingCFG


class Bridging4BlockCFG(BridgingCFG):
    MISSION_FILE = os.path.join(
        BridgingCFG.ROOT_DIR, "envs", "bridging_4block", "missions", "bridging_4block.xml"
    )
    GOAL_POS      = (0.5, 46.0, 6.5)
    Z_SUCCESS     = 6.0
    Z_SUCCESS_MAX = 8.0
    BRIDGE_Z_END  = 5      # gap is Z=2..5
    MAX_STEPS     = 250
