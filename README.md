# MalmoRL - Framework for Training RL Agents by extending Microsoft Malmo

A framework for training reinforcement learning agents in Minecraft using Microsoft Malmo. Define a task, pick an algorithm, train, and evaluate.

Built-in tasks include parkour (jumping across gaps) and block-bridging, but the framework is designed to support any Minecraft-based RL task.

**First time setup?** See [Setup & Installation](./Malmo/docs/framework/setup.md) before continuing.

## Documentation

### Framework

- [Setup & Installation](./Malmo/docs/framework/setup.md) — conda environments, Malmo installation, env vars
- [Observation Vector](./Malmo/docs/framework/observation_vector.md) — what the agent perceives on each step
- [Action Space](./Malmo/docs/framework/action_space.md) — available actions and how to modify them
- [Behavioral Cloning](./Malmo/docs/framework/behavioral_cloning.md) — demo recording, replay, BC + PPO pipeline
- [Curriculum Training](./Malmo/docs/framework/curriculum_training.md) — multi-env training with sequential, weighted, or adaptive progression
- [New Environments](./Malmo/docs/framework/new_environment.md) — how to add a new Malmo environment
- [New RL Algorithms](./Malmo/docs/framework/new_algorithm.md) — how to add a new training algorithm
- [New Model Architectures](./Malmo/docs/framework/new_model.md) — how to swap in a different network

### Parkour

- [Parkour Guide](./Malmo/docs/parkour/report.md) — world layout, observation space, action space, reward shaping, training pipeline

### Bridging

- [Bridging Guide](./Malmo/docs/bridging/bridging.md) — world layout, observation space, action space, reward shaping, training pipeline


## Extending the Framework to Custom Tasks

| What to add | Guide |
|-------------|-------|
| New task / environment | [New Environments](./Malmo/docs/framework/new_environment.md) — XML + config for new parkour variants; full env class for new task types |
| New RL algorithm | [New Algorithms](./Malmo/docs/framework/new_algorithm.md) — inherit from `BaseAgent`, implement 4 methods |
| New model architecture | [New Models](./Malmo/docs/framework/new_model.md) — implement 3 interface methods, swap in one import line |

---

## Quick Start — Bridging Agent

The bridging agent (crossing a 5-block gap by placing blocks) has a pre-trained baseline checkpoint, making it the fastest way to get started.

### 1. Launch Minecraft + Env Server

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Environment server
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
```

Wait for `Waiting for training script to connect...` before continuing.

### 2. Train (from the shared baseline checkpoint)

Start from scratch with demos → BC → PPO:

```powershell
# Record demos first
python Malmo/rl/utils/record_demos.py --env bridging --port 10002

# Then train: BC pre-training on demos, then PPO fine-tuning
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --demo-path Malmo/demos/bridging.json
```

Start from an existing baseline checkpoint (make sure --env flag is set correctly)
```powershell
# Terminal 3
conda activate train_env
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --checkpoint Malmo/rl/baselines/sb3_bridging_baseline.zip
```

### 3. Evaluate

```powershell
conda activate train_env
python Malmo/rl/evaluation/evaluate.py --env bridging --checkpoint checkpoints/sb3_bridging_final.zip --episodes 50 --port 10002
```

See the [Bridging Guide](./Malmo/docs/bridging/bridging.md) for full details.

---

## Built-in Environments

### Parkour

Jump across gaps of increasing difficulty. All parkour variants share the same env class — only the mission XML and config differ.

| Environment | Description |
|-------------|-------------|
| `one_block_gap` | 1-block gap, easiest |
| `simple_jump` | 2-block gap |
| `three_block_gap` | 3-block gap, hardest single jump |
| `vertical_small` | Forward + upward jump |
| `diagonal_small` | Forward + lateral jump |
| `diagonal_medium` | 3-block gap with lateral offset |
| `multi_jump_course` | 4-jump chained course |

### Bridging

Place blocks to build a bridge across an open gap. Requires inventory management and sneaking.

| Environment | Description |
|-------------|-------------|
| `bridging` | 5-block gap (main task) |
| `bridging_1block` | 1-block gap, easiest |
| `bridging_2block` | 2-block gap |
| `bridging_3block` | 3-block gap |
| `bridging_4block` | 4-block gap |

---

## Training Options

### Curriculum Training

Train across multiple environments in a single run with automatic progression:

```powershell
python Malmo/rl/training/train.py --curriculum path/to/curriculum.json --algo ppo --base-port 10002
```

See [Curriculum Training](./Malmo/docs/framework/curriculum_training.md) for the JSON format.

---

## Multi-Environment Training

Run N Minecraft clients in parallel for faster data collection:

```powershell
# Two Minecraft clients
cd .\Malmo\Minecraft && .\launchClient.bat
cd .\Malmo\Minecraft && .\launchClient.bat -port 10001

# Two env servers
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
python Malmo/rl/envs/env_server.py --env bridging --port 10003 --malmo-port 10001

# Train with 2 envs
conda activate train_env
python Malmo/rl/training/train_sb3.py --env bridging --num-envs 2 --base-port 10002
```
---



