# Malmo RL — Bridging Agent

A modular framework for training RL agents to bridge in Minecraft using Microsoft Malmo. The main focus is a block-bridging agent that learns to place blocks across a 5-block gap using behavioral cloning from human demos followed by PPO fine-tuning.

**First time setup?** See [Setup & Installation](./Malmo/docs/setup.md) before continuing.

---

## Quick Start — Bridging Agent

Training the bridging agent takes 4 steps. You need 2 terminals running at all times (Minecraft client + env server), with a 3rd for whichever task you're doing.

### 1. Launch Minecraft + Env Server

Keep these running throughout recording, training, and evaluation.

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Environment server
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
```

Wait for `Waiting for training script to connect...` before proceeding.

### 2. Record Human Demonstrations

Record 50+ successful episodes for behavioral cloning.

```powershell
# Terminal 3
conda activate train_env
python Malmo/rl/utils/record_demos.py --env bridging --port 10002
```

**Controls:**

| Key | Action |
|-----|--------|
| W / S | Forward / backward |
| A / D | Strafe left / right |
| Shift | Sneak (hold) |
| Right-click | Place block |
| Shift+W | Sneak forward |
| Shift+Right-click | Sneak + place |
| Arrow keys | Look / turn |
| Esc | Save & quit |

Demos save to `Malmo/demos/bridging.json` and append across sessions. To start fresh: `rm demos/bridging.json`

**Replay demos to verify quality** (env server must be running):

```powershell
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002 --episode 0 --speed 0.5
```

### 3. Train

Two training styles are available: **single-env** (simple, uses SB3) and **curriculum** (recommended, uses custom PPO).

#### Option A — Curriculum training (recommended)

Starts on a 1-block gap and auto-promotes to larger gaps once 90% success rate is reached over a 50-episode window. Uses BC pre-training first, then PPO fine-tuning.

**Step 1 — BC pre-train on demos** (offline, no Minecraft needed):
```powershell
# Terminal 3
conda activate train_env
python Malmo/rl/training/train.py --env bridging_1block --algo bc --demo-path Malmo/demos/bridging.json --base-port 10002
```
This runs quickly (minutes). A checkpoint is saved to `checkpoints/bc_bridging_1block_ep<N>.pt`.

**Step 2 — PPO curriculum from BC checkpoint** (env server must be running on `bridging_1block`):
```powershell
# Terminal 2: restart env server on bridging_1block
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging_1block --port 10002 --malmo-port 10000

# Terminal 3
conda activate train_env
python Malmo/rl/training/train.py --algo ppo --curriculum Malmo/rl/training/curricula/adaptive_bridging.json --checkpoint Malmo/rl/checkpoints/bc_bridging_1block_ep<N>.pt --base-port 10002
```
The curriculum handles env switching automatically (1→2→3→4→5 block gaps). The env server only needs to start on the first env; `train.py` sends `switch_env` commands as the agent promotes.

**Resume curriculum from a PPO checkpoint:**
```powershell
python Malmo/rl/training/train.py --algo ppo --curriculum Malmo/rl/training/curricula/adaptive_bridging.json --checkpoint Malmo/rl/checkpoints/ppo_curriculum_ppo_ep<N>.pt --base-port 10002
```

#### Option B — Single-env SB3 training

```powershell
# Terminal 3

# BC pre-training on demos, then PPO fine-tuning:
conda activate train_env
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --demo-path demos/bridging.json

# PPO from scratch (no demos):
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002

# Resume from checkpoint:
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --checkpoint checkpoints/sb3_bridging_100000_steps.zip
```

### 4. Evaluate

```powershell
conda activate train_env
python Malmo/rl/evaluation/evaluate.py --env bridging --checkpoint checkpoints/sb3_bridging_final.zip --episodes 50 --port 10002
```

---

## Multi-Environment Training

Run N Minecraft clients for faster data collection. Each needs its own client + env server.

```powershell
# Terminal 1: Minecraft client 1 (Malmo port 10000)
cd .\Malmo\Minecraft && .\launchClient.bat
# Terminal 2: Minecraft client 2 (Malmo port 10001)
cd .\Malmo\Minecraft && .\launchClient.bat -port 10001

# Terminal 3: env server 1
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
# Terminal 4: env server 2
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10003 --malmo-port 10001

# Terminal 5: training with 2 envs
conda activate train_env
python Malmo/rl/training/train_sb3.py --env bridging --num-envs 2 --base-port 10002
```

> **Windows note:** Some port ranges may be reserved by Hyper-V. If you get `WinError 10013`, pick a different `--port`. Check reserved ranges with `netsh interface ipv4 show excludedportrange protocol=tcp`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ConnectionRefusedError` | Start Minecraft + env server before running any script |
| `WinError 10013` on env server | Port reserved by Hyper-V — pick a different `--port` |
| Mission ends before success | Increase `timeLimitMs` in `bridging.xml` (currently 120000ms) |
| Previous mission still running | Wait ~30s or restart Minecraft client |

---

## Documentation

- [Bridging Agent Full Guide](./malmo/docs/bridging_quickstart.md) — world layout, action space, reward structure, key files
- [Behavioral Cloning](./malmo/docs/behavioral_cloning.md) — demo recording, replay, BC + PPO pipeline
- [Setup & Installation](./malmo/docs/setup.md) — conda environments, Malmo installation, env vars
- [New Environments](./malmo/docs/new_environment.md) — how to add a new Malmo environment
- [New RL Algorithms](./malmo/docs/new_algorithm.md) — how to add a new training algorithm
- [New Model Architectures](./malmo/docs/new_model.md) — how to swap in a different network
- [Observation Vector](./malmo/docs/observation_vector.md) — what the agent perceives on each step
- [Action Space](./malmo/docs/action_space.md) — available actions and how to modify them
- [Architecture Report](./malmo/docs/parkour_bot_report.md) — full system design and decisions
