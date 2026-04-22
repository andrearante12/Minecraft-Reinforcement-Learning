import os
from training.configs.bridging_cfg import BridgingCFG


class Bridging3BlockCFG(BridgingCFG):
    MISSION_FILE = os.path.join(
        BridgingCFG.ROOT_DIR, "envs", "bridging_3block", "missions", "bridging_3block.xml"
    )
    GOAL_POS      = (0.5, 46.0, 5.5)
    Z_SUCCESS     = 5.0
    Z_SUCCESS_MAX = 7.0
    BRIDGE_Z_END  = 4      # gap is Z=2..4
    MAX_STEPS     = 200
