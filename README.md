# Malmo RL Platform

A modular framework for training and evaluating reinforcement learning agents in Minecraft using Microsoft Malmo.

## Documentation

- [Setup & Installation](./malmo/docs/setup.md) — conda environments, Malmo installation, environment variables
- [Observation Vector](./malmo/docs/observation_vector.md) — what the agent perceives on each step
- [Action Space](./malmo/docs/action_space.md) — available actions and how to modify them
- [New Model Architectures](./malmo/docs/new_model.md) — how to swap in a different network
- [New RL Algorithms](./malmo/docs/new_algorithm.md) — how to add a new training algorithm
- [New Environments](./malmo/docs/new_environment.md) — how to create a new Malmo environment

---

## Running the Agent

**First time setup?** See [Setup & Installation](./malmo/docs/setup.md) before continuing.

### Single Environment (3 terminals)

**1. Launch the Minecraft client** (leave this running):

```powershell
cd .\Malmo\Minecraft\
.\launchClient.bat
```

**2. Start the environment server** (Terminal 2):

```powershell
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump
```

To run a different environment, change the `--env` flag:

```powershell
python Malmo/rl/envs/env_server.py --env three_block_gap
```

**3. Start training** (Terminal 3):

```powershell
conda activate train_env
python Malmo/rl/training/train.py --env simple_jump --algo ppo
```

### Multiple Parallel Environments

Running N environments simultaneously collects N transitions per step, filling the PPO rollout buffer N times faster with more diverse data. Each environment needs its own Minecraft client and env server.

**1. Launch N Minecraft clients** on separate Malmo ports (one terminal each):

```powershell
cd .\Malmo\Minecraft\
.\launchClient.bat                # Client 1 — default Malmo port 10000
.\launchClient.bat -port 10001    # Client 2 — Malmo port 10001
```

**2. Start N environment servers**, each targeting its own Minecraft client and listening on its own TCP port:

```powershell
conda activate malmo

# Env server 1: Minecraft on port 10000, TCP on port 10002
python Malmo/rl/envs/env_server.py --env simple_jump --port 10002 --malmo-port 10000

# Env server 2: Minecraft on port 10001, TCP on port 10003
python Malmo/rl/envs/env_server.py --env simple_jump --port 10003 --malmo-port 10001
```

**3. Start training** with `--num-envs` and `--base-port`:

```powershell
conda activate train_env
python Malmo/rl/training/train.py --env simple_jump --algo ppo --num-envs 2 --base-port 10002
```

The training script connects to env servers at `base-port`, `base-port+1`, ..., `base-port+N-1`.

> **Windows note:** Some TCP port ranges (e.g. 10000–10001) may be reserved by Hyper-V. If you get `WinError 10013` when starting an env server, choose a different `--port`. Run `netsh interface ipv4 show excludedportrange protocol=tcp` to see reserved ranges.

---

## Useful Commands

Resume training from a checkpoint:

```powershell
python Malmo/rl/training/train.py --env simple_jump --algo ppo --checkpoint rl/checkpoints/ppo_simple_jump_ep500.pt
```

Evaluate a trained checkpoint:

```powershell
conda activate train_env
python Malmo/rl/evaluation/evaluate.py --checkpoint rl/checkpoints/ppo_simple_jump_ep1000.pt
python Malmo/rl/evaluation/evaluate.py --checkpoint rl/checkpoints/ppo_simple_jump_ep1000.pt --episodes 50
```
