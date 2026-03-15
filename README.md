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

**1. Launch the Minecraft client** (leave this running):

```powershell
cd .\Malmo\Minecraft\
.\launchClient.bat
```

**2. Start the environment server** (Terminal 1):

```powershell
conda activate malmo
python Malmo/parkour/envs/env_server.py --env simple_jump
```

To run a different environment, change the `--env` flag:

```powershell
python Malmo/parkour/envs/env_server.py --env three_block_gap
```

**3. Start training** (Terminal 2):

```powershell
conda activate train_env
python Malmo/parkour/training/train.py --env simple_jump --algo ppo
```

---

## Useful Commands

Resume training from a checkpoint:

```powershell
python Malmo/parkour/training/train.py --env simple_jump --algo ppo --checkpoint parkour/checkpoints/ppo_simple_jump_ep500.pt
```

Evaluate a trained checkpoint:

```powershell
conda activate train_env
python Malmo/parkour/evaluation/evaluate.py --checkpoint parkour/checkpoints/ppo_simple_jump_ep1000.pt
python Malmo/parkour/evaluation/evaluate.py --checkpoint parkour/checkpoints/ppo_simple_jump_ep1000.pt --episodes 50
```