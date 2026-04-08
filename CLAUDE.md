# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular framework for training RL agents to do parkour in Minecraft using Microsoft Malmo. The architecture requires **two separate Python environments** (Malmo requires Python 3.7, PyTorch requires 3.10+) communicating over TCP.

## Commands

### Environment Setup (one-time)
```powershell
conda env create -f .\conda_environments\malmo_environment.yml      # Python 3.7, runs Malmo
conda env create -f .\conda_environments\training_environment.yml   # Python 3.10, runs PyTorch

# Copy Malmo Python binding into the malmo conda env
copy ".\Malmo\Python_Examples\MalmoPython.pyd" "$env:CONDA_PREFIX\Lib\site-packages\"

# Set system env vars (as Administrator)
[System.Environment]::SetEnvironmentVariable("MALMO_XSD_PATH", "<repo>\Malmo\Schemas", "Machine")
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", "<repo>\Malmo\rl", "User")
```

### Running Training (3 terminals required)
```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Environment server (malmo env, Python 3.7)
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump

# Terminal 3: Training (train_env, Python 3.10)
conda activate train_env
python Malmo/rl/training/train.py --env simple_jump --algo ppo
python Malmo/rl/training/train.py --env simple_jump --algo ppo --checkpoint checkpoints/ppo_simple_jump_ep500.pt
```

### Multi-Env Training (N=2 example)
Each env server needs its own Minecraft client and its own Malmo port.
```powershell
# Terminal 1: Minecraft client 1 (default Malmo port 10000)
cd .\Malmo\Minecraft && .\launchClient.bat
# Terminal 2: Minecraft client 2 (Malmo port 10001)
cd .\Malmo\Minecraft && .\launchClient.bat -port 10001

# Terminal 3: env server 1 → Minecraft on port 10000, TCP on port 10002
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump --port 10002 --malmo-port 10000
# Terminal 4: env server 2 → Minecraft on port 10001, TCP on port 10003
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump --port 10003 --malmo-port 10001

# Terminal 5: training connects to TCP ports 10002, 10003
conda activate train_env
python Malmo/rl/training/train.py --env simple_jump --algo ppo --num-envs 2 --base-port 10002
```
Note: On Windows, some port ranges (e.g. 10000-10001) may be reserved by Hyper-V for TCP servers.
If you hit `WinError 10013`, pick a different `--base-port`.
Use `netsh interface ipv4 show excludedportrange protocol=tcp` to check reserved ranges.

### Curriculum Training
```powershell
# Same Minecraft clients + env servers as multi-env training above.
# The training script handles env switching via --curriculum:
conda activate train_env
python Malmo/rl/training/train.py --curriculum path/to/curriculum.json --algo ppo --base-port 10002

# Resume from checkpoint:
python Malmo/rl/training/train.py --curriculum path/to/curriculum.json --algo ppo --checkpoint checkpoints/ppo_curriculum_ppo_ep500.pt
```
See `Malmo/docs/curriculum_training.md` for JSON format and details.

### Behavioral Cloning (Demo Recording + Pre-training)
```powershell
# Record human demonstrations (env server must be running):
conda activate train_env
python Malmo/rl/utils/record_demos.py --env one_block_gap --port 10002

# BC pre-training on demo data:
python Malmo/rl/training/train.py --env one_block_gap --algo bc --demo-path demos/one_block_gap.json --base-port 10002

# PPO fine-tuning from BC checkpoint:
python Malmo/rl/training/train.py --env one_block_gap --algo ppo --checkpoint checkpoints/ppo_one_block_gap_bc_ep500.pt --base-port 10002

# Replay recorded demos in Minecraft (env server must be running):
python Malmo/rl/utils/replay_demos.py --env one_block_gap --port 10002
python Malmo/rl/utils/replay_demos.py --env one_block_gap --port 10002 --episode 0 --speed 0.5

# Clear demos for an environment:
rm demos/one_block_gap.json
```
See `Malmo/docs/behavioral_cloning.md` for full guide.

### Evaluation
```powershell
# Requires env_server.py running on the target env:
python Malmo/rl/envs/env_server.py --env simple_jump --port 10002 --malmo-port 10000

conda activate train_env
python Malmo/rl/evaluation/evaluate.py --env simple_jump --checkpoint checkpoints/ppo_simple_jump_ep1000.pt --episodes 50 --port 10002

# Multi-jump course evaluation:
python Malmo/rl/evaluation/evaluate.py --env multi_jump_course --checkpoint <checkpoint>.pt --episodes 50 --port 10002
```

## Architecture

### Two-Process Design
The core constraint: Malmo's Python bindings only work with Python 3.7, while modern PyTorch requires 3.10+. The solution is a TCP server/client split:

```
malmo (Python 3.7)                         train_env (Python 3.10)
─────────────────────────────────          ─────────────────────────
env_server.py (TCP :9999)          ←→      train.py
  └── envs/parkour_env.py                    ├── envs/env_client.py
        └── MalmoPython + Minecraft JVM      ├── models/mlp.py
                                             ├── algos/ppo.py or dqn.py
                                             └── utils/logger.py
```

Communication is JSON over TCP: `{"cmd": "reset"/"step", "action": int}` → `{"obs": [...], "reward": float, "done": bool, "info": {...}}`

### Observation Space (159-dim vector)
- **[0-5]**: Proprioception — onGround, yaw, pitch, delta_y, delta_x, delta_z
- **[6-8]**: Goal delta — goal_dx, goal_dy, goal_dz
- **[9-158]**: Voxel grid — 5×5×6=150 blocks encoded as 0=air, 1=stone

### Action Space (15 discrete)
forward, backward, left, right, sprint_forward, jump, sprint_jump, jump_forward, sprint_jump_left, sprint_jump_right, look_down, look_up, turn_left, turn_right, no_op

### Model (`models/mlp.py`)
`ActorCritic`: two shared Linear(→128)→Tanh layers, then separate policy head (→Categorical) and value head (→scalar).

### Extensibility — Registry Pattern
- `env_server.py` has `ENV_REGISTRY`: maps env name → `(EnvClass, CfgClass)` tuple
- `train.py` has `ENV_REGISTRY` (same format) and `ALGO_REGISTRY`: maps algo name → agent class
- `CurriculumScheduler` (`training/curriculum.py`): schedules env switching across episodes (sequential stages or weighted random sampling). Validated to ensure all envs share `INPUT_SIZE`/`N_ACTIONS`.
- Adding a new environment, algorithm, or model only requires implementing an interface and registering it — the training loop, logging, and checkpointing require no changes.

### Adding a New Environment
1. Create `Malmo/rl/envs/<name>/missions/<name>.xml` (copy existing XML, modify geometry)
2. Create `Malmo/rl/training/configs/<name>_cfg.py` (copy existing config, adjust values including `SUCCESS_REQUIRES_ON_GROUND` and `REWARD_ON_MISSION_ENDED` if needed)
3. Register config in `ENV_REGISTRY` in `env_server.py` and in `train.py`

### Adding a New Algorithm
1. Create `Malmo/rl/algos/<name>.py` inheriting `BaseAgent`
2. Implement: `collect_step()`, `update()`, `select_action()`, optionally `buffer_full()`, `_extra_state()`, `_load_extra_state()`
3. Register in `ALGO_REGISTRY` in `train.py`

### Key Hyperparameters (`training/configs/base_cfg.py`)
PPO defaults: N_STEPS=512, N_EPOCHS=4, CLIP_EPS=0.2, ENTROPY_COEF=0.05, LR=3e-4
DQN defaults: BUFFER_CAPACITY=10000, EPSILON_START=1.0, EPSILON_END=0.05, TARGET_UPDATE_FREQ=500
BC defaults: BC_EPOCHS=10, BC_BATCH_SIZE=64, DEMO_PATH=None (set via --demo-path)
Rewards: SUCCESS=+10, FELL=-5, TIMEOUT=-5, STEP_PENALTY=-0.01
Landing phase: LANDING_TICKS=0 (instant success) or >0 (must stay on block), REWARD_LANDING_TICK=+0.5, Z_SUCCESS_MAX=None

### Logging
Each run produces two timestamped CSVs in `Malmo/rl/logs/`:
- `*_episodes.csv`: episode, reward, steps, outcome, env
- `*_updates.csv`: update number + dynamic loss columns (automatically adapts to algorithm)

Checkpoints saved every 100 episodes to `Malmo/rl/checkpoints/`. Interrupted training saves `*_interrupted.pt`.

## Documentation
- `Malmo/docs/setup.md` — full installation guide
- `Malmo/docs/observation_vector.md` — observation space details
- `Malmo/docs/action_space.md` — action space and Malmo command reference
- `Malmo/docs/new_algorithm.md`, `new_environment.md`, `new_model.md` — extension guides
- `Malmo/docs/curriculum_training.md` — curriculum training guide
- `Malmo/docs/behavioral_cloning.md` — BC + PPO fine-tuning pipeline
- `Malmo/docs/parkour_bot_report.md` — full architecture report and design decisions
