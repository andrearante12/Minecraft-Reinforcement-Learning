# Creating a New Environment

This guide covers two scenarios:

1. **New parkour variant** — same observation/reward structure, different geometry (e.g. wider gap, higher platform). Only requires an XML file, a config, and two registry entries.
2. **New task type** — fundamentally different gameplay like gathering materials or combat. Requires a new env class with its own observation space, action space, and reward logic.

---

## Part A — New Parkour Variant

All parkour environments share a single `ParkourEnv` class (`envs/parkour_env.py`). Behavioral differences are driven entirely by config flags. This walkthrough uses **three-block gap** as a worked example.

### Architecture

```
malmo env (Python 3.7)              train_env (Python 3.10)
──────────────────────              ──────────────────────
envs/env_server.py                  training/train.py
  └── envs/parkour_env.py ←─JSON─→ envs/env_client.py
        MalmoPython                   PPO / DQN
```

`env_server.py` maps the `--env` name to a config class and passes it to `ParkourEnv`. No per-environment env.py files are needed.

### Step 1 — Create the folder structure

```powershell
mkdir rl\envs\three_block_gap\missions
```

### Step 2 — Create the mission XML

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

### Step 3 — Create the config file

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

    # ── Behavior overrides (defaults in BaseCFG) ────────────────────────────
    SUCCESS_REQUIRES_ON_GROUND = True
    REWARD_ON_MISSION_ENDED    = False
```

#### Config behavior flags

Two flags in `BaseCFG` control the behavioral differences between parkour variants:

| Flag | Default | Effect |
|------|---------|--------|
| `SUCCESS_REQUIRES_ON_GROUND` | `False` | `True` = success requires `OnGround` flag. `False` = success requires `y >= FALL_Y_THRESHOLD` |
| `REWARD_ON_MISSION_ENDED` | `True` | `True` = assign `REWARD_TIMEOUT` when mission ends unexpectedly. `False` = no reward penalty |

Override these in your config only if the defaults don't match your env's needs.

### Step 4 — Register in both registries

**`envs/env_server.py`** — add the config import and registry entry:

```python
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG

ENV_REGISTRY = {
    "simple_jump":     SimpleJumpCFG,
    "three_block_gap": ThreeBlockGapCFG,  # add this
}
```

**`training/train.py`** — same pattern:

```python
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG

ENV_REGISTRY = {
    "simple_jump":     SimpleJumpCFG,
    "three_block_gap": ThreeBlockGapCFG,  # add this
}
```

### Step 5 — Run it

Terminal 1 (Minecraft client):
```powershell
cd .\Malmo\Minecraft && .\launchClient.bat
```

Terminal 2 (malmo env):
```powershell
conda activate malmo
python Malmo/rl/envs/env_server.py --env three_block_gap
```

Terminal 3 (train_env):
```powershell
conda activate train_env
python Malmo/rl/training/train.py --env three_block_gap --algo ppo
```

### Parkour variant checklist

| What | Where |
|------|-------|
| Mission XML | `envs/<name>/missions/<name>.xml` |
| Config file | `training/configs/<name>_cfg.py` |
| Register in server | `envs/env_server.py` — one import + one registry entry |
| Register in train | `training/train.py` — one import + one registry entry |

That's it — no env.py needed. `ParkourEnv` handles everything.

---

## Part B — New Task Type (Non-Parkour)

If the task has fundamentally different gameplay — gathering materials, combat, navigation to waypoints — you need a custom env class. The parkour observation space (proprioception + goal delta + voxel grid) and reward logic (fall detection, z-position success) won't apply.

### What changes

| Component | Parkour variant | New task type |
|-----------|----------------|---------------|
| Mission XML | New file | New file |
| Config | Inherits `BaseCFG`, overrides geometry | Inherits `BaseCFG`, defines new task-specific fields |
| Env class | Reuses `ParkourEnv` | New class (e.g. `CombatEnv`, `GatherEnv`) |
| Observation space | 129-dim (fixed) | Custom — depends on what the agent needs to perceive |
| Action space | 12 discrete movement actions | Custom — may need attack, use item, inventory, etc. |
| Reward function | Position-based (fell/landed/progress) | Custom — damage dealt, items collected, etc. |
| `env_server.py` | Config-only registry entry | Needs both env class and config in registry |

### Step 1 — Design the observation and action spaces

Before writing code, decide:

- **What does the agent observe?** The Malmo XML `<AgentHandlers>` section controls what data is available. Common handlers beyond the parkour defaults:
  - `<ObservationFromNearbyEntities>` — nearby mobs/players (combat)
  - `<ObservationFromHotBar>` / `<InventoryCommands>` — inventory state (gathering)
  - `<ObservationFromRay>` — what the agent is looking at (interaction tasks)
  - `<ObservationFromRecentCommands>` — action feedback
  - See [Malmo XML schema docs](http://microsoft.github.io/malmo/0.30.0/Schemas/MissionHandlers.html) for the full list

- **What actions can the agent take?** Malmo command handlers to consider:
  - `<ContinuousMovementCommands>` — movement (already used in parkour)
  - `<DiscreteMovementCommands>` — grid-based movement
  - `<InventoryCommands>` — swap/select inventory slots
  - `<SimpleCraftCommands>` — crafting
  - `<ChatCommands>` — sending chat (useful for some Malmo built-in commands)

- **`INPUT_SIZE`** and **`N_ACTIONS`** must be defined in your config — the model and training loop depend on them.

### Step 2 — Create the mission XML

Same process as parkour, but the `<AgentHandlers>` section will look different. Example for a gathering task:

```xml
<AgentHandlers>
  <ContinuousMovementCommands/>
  <InventoryCommands/>
  <ObservationFromFullStats/>
  <ObservationFromHotBar/>
  <ObservationFromGrid>
    <Grid name="surroundings">
      <min x="-3" y="-1" z="-3"/>
      <max x="3"  y="2"  z="3"/>
    </Grid>
  </ObservationFromGrid>
  <ObservationFromNearbyEntities>
    <Range name="nearby_items" xrange="10" yrange="2" zrange="10"/>
  </ObservationFromNearbyEntities>
</AgentHandlers>
```

### Step 3 — Create the config

Inherit from `BaseCFG` for shared training hyperparameters (learning rate, PPO/DQN settings, etc.), and define task-specific fields:

```python
import os
from training.configs.base_cfg import BaseCFG


class GatherWoodCFG(BaseCFG):
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "gather_wood", "missions", "gather_wood.xml"
    )
    MALMO_PORT = 10000

    SPAWN    = (0.5, 4.0, 0.5)
    MAX_STEPS    = 500
    STEP_DURATION = 0.15

    # ── Task-specific rewards ────────────────────────────────────────────────
    REWARD_COLLECT_WOOD = +5.0
    REWARD_STEP_PENALTY = -0.05

    # ── Observation space ────────────────────────────────────────────────────
    # Define whatever makes sense for your task
    INPUT_SIZE = 200   # example — must match what your env class builds

    # ── Action space ─────────────────────────────────────────────────────────
    ACTIONS = [
        ("move_forward",  ["move 1"],    ["move 0"]),
        ("turn_left",     ["turn -1"],   ["turn 0"]),
        ("turn_right",    ["turn 1"],    ["turn 0"]),
        ("jump",          ["jump 1"],    ["jump 0"]),
        ("attack",        ["attack 1"],  ["attack 0"]),
        ("no_op",         [],            []),
    ]
    N_ACTIONS = len(ACTIONS)
```

### Step 4 — Create the env class

Create a new file like `envs/gather_env.py`. Your env class must implement the same interface that `env_server.py` and `env_client.py` expect:

```python
class GatherEnv:
    def __init__(self, cfg, malmo_port=None):
        """Set up Malmo agent host, load mission XML, init state."""
        ...

    def reset(self):
        """Start a new episode. Returns obs (np.ndarray of shape (INPUT_SIZE,))."""
        ...

    def step(self, action):
        """Execute action. Returns (obs, reward, done, info)."""
        ...
        # info must be a dict with at least: outcome, steps, pos, action
        ...

    def close(self):
        """Clean up."""
        ...
```

**Key constraints:**
- `reset()` returns a numpy array of shape `(cfg.INPUT_SIZE,)`
- `step()` returns `(obs, reward, done, info)` where `info` is a JSON-serializable dict
- The `info` dict should include `"outcome"` (string), `"steps"` (int), `"pos"` (tuple), and `"action"` (string) — the logger and evaluator use these fields

You can use `envs/parkour_env.py` as a reference for the Malmo boilerplate (`_start_mission`, `_take_action`, `_get_obs_dict`). Copy those methods and customize `_build_obs_vector`, `_get_reward`, and the action handling.

### Step 5 — Register in `env_server.py`

For non-parkour envs, the registry needs both the env class and the config. Update `env_server.py` to support this:

```python
from envs.gather_env import GatherEnv
from training.configs.gather_wood_cfg import GatherWoodCFG

ENV_REGISTRY = {
    # Parkour envs — config-only (ParkourEnv is used automatically)
    "simple_jump":     SimpleJumpCFG,
    "three_block_gap": ThreeBlockGapCFG,
    # Non-parkour envs — (EnvClass, CfgClass) tuples
    "gather_wood":     (GatherEnv, GatherWoodCFG),
}
```

Then update the `main()` function to handle both formats:

```python
entry = ENV_REGISTRY[args.env]
if isinstance(entry, tuple):
    EnvClass, cfg = entry
else:
    EnvClass, cfg = ParkourEnv, entry
env = EnvClass(cfg, malmo_port=args.malmo_port)
```

### Step 6 — Register in `train.py`

`train.py` only needs the config (it never imports env classes — it talks to the env through `EnvClient` over TCP):

```python
from training.configs.gather_wood_cfg import GatherWoodCFG

ENV_REGISTRY = {
    "simple_jump":     SimpleJumpCFG,
    "gather_wood":     GatherWoodCFG,   # add this
}
```

### Step 7 — Consider model changes

The default `ActorCritic` model in `models/mlp.py` takes `INPUT_SIZE` and `N_ACTIONS` from the config. If your observation is a flat vector, the existing model works as-is — just set `INPUT_SIZE` and `N_ACTIONS` correctly in your config.

If your task needs a fundamentally different architecture (e.g. CNN for image observations, attention for variable-length entity lists), create a new model in `models/` and register it in `train.py`. See `Malmo/docs/new_model.md`.

### Non-parkour checklist

| What | Where |
|------|-------|
| Mission XML | `envs/<name>/missions/<name>.xml` |
| Config file | `training/configs/<name>_cfg.py` (inherit `BaseCFG`) |
| Env class | `envs/<name>_env.py` (implement `reset`, `step`, `close`) |
| Register in server | `envs/env_server.py` — `(EnvClass, CfgClass)` tuple in registry |
| Register in train | `training/train.py` — config-only registry entry |
| Model (if needed) | `models/<name>.py` — only if default MLP doesn't fit |
