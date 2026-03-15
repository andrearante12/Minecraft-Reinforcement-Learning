# Creating a New Environment

This guide walks through adding a new environment from scratch using the **three-block gap** as a real worked example — identical to `simple_jump` but with a gap one block wider.

By the end you will have a fully working environment runnable with:
```powershell
python parkour/envs/env_server.py --env three_block_gap
python parkour/training/train.py --env three_block_gap --algo ppo
```

---

## How the Environment System Works

The environment system is split into two processes connected by a TCP socket. This is required because Malmo (Python 3.7) and PyTorch (Python 3.10) cannot run in the same Python environment.

```
malmo env (Python 3.7)              train_env (Python 3.10)
──────────────────────              ──────────────────────
envs/env_server.py                  training/train.py
  └── envs/<name>/env.py ←─JSON─→  envs/env_client.py
        MalmoPython                   PPO / DQN
```

`env_server.py` and `env_client.py` are **shared across all environments** — you never create per-environment versions of them. The only files you need to create for a new environment are:

```
envs/
└── three_block_gap/
    ├── __init__.py
    ├── env.py                         ← Malmo wrapper (copy + change one line)
    └── missions/
        └── three_block_gap.xml        ← world definition

training/configs/
└── three_block_gap_cfg.py             ← env-specific settings
```

Then two lines each in `env_server.py` and `train.py` to register it.

---

## Step 1 — Create the folder structure

```powershell
mkdir parkour\envs\three_block_gap
mkdir parkour\envs\three_block_gap\missions
New-Item -Path "parkour\envs\three_block_gap\__init__.py" -ItemType File
```

---

## Step 2 — Create the mission XML

Save as `envs/three_block_gap/missions/three_block_gap.xml`:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>Three-Block Gap Challenge</Summary>
  </About>

  <ServerSection>
    <ServerInitialConditions>
      <Time><StartTime>1000</StartTime></Time>
      <Weather>clear</Weather>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2*3,2;1;" forceReset="true"/>
      <DrawingDecorator>
        <!-- Starting platform: Z -20 to Z 3 -->
        <DrawCuboid x1="0" y1="45" z1="-20" x2="0" y2="45" z2="3" type="stone"/>
        <!-- Landing platform: Z 7 to Z 20 (gap is Z 4, 5, 6 = 3 blocks wide) -->
        <DrawCuboid x1="0" y1="45" z1="7" x2="0" y2="45" z2="20" type="stone"/>
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="30000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Parkour Agent</Name>
    <AgentStart>
      <Placement x="0.5" y="46" z="0.5" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
      <ContinuousMovementCommands/>
      <ObservationFromFullStats/>
      <ObservationFromGrid>
        <Grid name="floor3x3">
          <min x="-2" y="-1" z="-2"/>
          <max x="2"  y="2"  z="3"/>
        </Grid>
      </ObservationFromGrid>
    </AgentHandlers>
  </AgentSection>
</Mission>
```

**Important XML notes:**
- `forceReset="true"` stays in the XML — the world rebuilds on each episode reset which guarantees a clean respawn. Do not change this to `false`
- `mode="Survival"` is intentional — when the agent falls they die and Malmo ends the mission cleanly, respawning them at `<Placement>` on the next episode start

---

## Step 3 — Create the config file

Save as `training/configs/three_block_gap_cfg.py`:

```python
import os
from training.configs.base_cfg import BaseCFG


class ThreeBlockGapCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "three_block_gap", "missions", "three_block_gap.xml"
    )
    MALMO_PORT = 10000

    SPAWN    = (0.5, 46.0, 0.5)
    GOAL_POS = (0.5, 45.0, 8.0)

    FALL_Y_THRESHOLD = 43.0
    Z_SUCCESS        = 7.0         # landing platform starts at Z=7
    MAX_STEPS        = 200
    STEP_DURATION    = 0.15

    GRID_X    = 5
    GRID_Y    = 4
    GRID_Z    = 6
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 120

    BLOCK_ENCODING = {"air": 0, "stone": 1}
    GOAL_BLOCK     = "stone"

    PROPRIOCEPTION_SIZE = 6
    GOAL_DELTA_SIZE     = 3
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE  # 129

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
```

**What changed vs `simple_jump_cfg.py`:**

| Setting | simple_jump | three_block_gap | Why |
|---------|-------------|-----------------|-----|
| `MISSION_FILE` | `simple_jump/...` | `three_block_gap/...` | Points to new XML |
| `GOAL_POS` | `(0.5, 45.0, 7.0)` | `(0.5, 45.0, 8.0)` | Platform center one block further |
| `Z_SUCCESS` | `5.0` | `7.0` | Platform starts at Z=7 |

`INPUT_SIZE` stays 129 — the observation layout is unchanged.

---

## Step 4 — Create `env.py`

Copy `simple_jump/env.py` and change the one config import line:

```powershell
copy parkour\envs\simple_jump\env.py parkour\envs\three_block_gap\env.py
```

Open the copy and change the config import:

```python
# Before
from training.configs.simple_jump_cfg import SimpleJumpCFG as CFG

# After
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG as CFG
```

That is the only change. The full `env.py` for reference:


---

## Step 5 — Register in `env_server.py`

Add two lines to `envs/env_server.py`:

```python
# Add import
from envs.three_block_gap.env import ParkourEnv as ThreeBlockGapEnv
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG

# Add to registry
ENV_REGISTRY = {
    "simple_jump":     (SimpleJumpEnv,    SimpleJumpCFG),
    "three_block_gap": (ThreeBlockGapEnv, ThreeBlockGapCFG),  # add this
}
```

---

## Step 6 — Register in `train.py`

Add two lines to `training/train.py`:

```python
# Add import
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG

# Add to registry
ENV_REGISTRY = {
    "simple_jump":     SimpleJumpCFG,
    "three_block_gap": ThreeBlockGapCFG,  # add this
}
```

---

## Step 7 — Run it

Terminal 1 (malmo env):
```powershell
conda activate malmo
python parkour/envs/env_server.py --env three_block_gap
```

Terminal 2 (train_env):
```powershell
conda activate train_env
python parkour/training/train.py --env three_block_gap --algo ppo
```

---

## Summary — what you need for every new environment

| What | Where | Notes |
|------|-------|-------|
| `__init__.py` | `envs/<name>/__init__.py` | Empty, required by Python |
| Mission XML | `envs/<name>/missions/<name>.xml` | Use `<Name>` not `<n>`, keep `forceReset="true"` and `mode="Survival"` |
| Config file | `training/configs/<name>_cfg.py` | Copy existing cfg, update geometry values |
| `env.py` | `envs/<name>/env.py` | Copy from `simple_jump/env.py`, change one import line |
| Register in server | `envs/env_server.py` | Two lines in `ENV_REGISTRY` |
| Register in train | `training/train.py` | Two lines in `ENV_REGISTRY` |

---
