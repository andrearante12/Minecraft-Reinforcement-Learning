# MalmoRL — Project Context for Claude

This file is shared baseline context for Claude Code and all sub-agents in this repo.

## What this project is

MalmoRL is a modular framework for training reinforcement-learning agents in Minecraft using Microsoft Malmo. Users compose experiments by mixing **environments**, **reward functions**, **algorithms**, and **model architectures**, all wired together through small registries. Built-in tasks include parkour (jumping across gaps) and bridging (placing blocks to cross a gap).

## How to extend it (the registry model)

Every extension follows the same pattern:

| What you're adding | Files you write | Registries you update |
|---|---|---|
| **Parkour variant** (geometry only) | `malmo/rl/envs/<name>/missions/<name>.xml`, `malmo/rl/training/configs/<name>_cfg.py` | `ENV_REGISTRY` in **both** `malmo/rl/envs/env_server.py` **and** `malmo/rl/training/train.py` |
| **New task type** (new env class) | mission XML + cfg + `malmo/rl/envs/<name>_env.py` | same two `ENV_REGISTRY` blocks |
| **New algorithm** | `malmo/rl/algos/<name>.py` inheriting `BaseAgent` | `ALGO_REGISTRY` in `malmo/rl/training/train.py` |
| **New model** | `malmo/rl/models/<name>.py` implementing 3 methods | model import line in `malmo/rl/training/train.py` |
| **Reward tuning** | edit `<env>_cfg.py` constants and/or `_get_reward()` in `<env>_env.py` | none |

**Footgun #1 (most common newcomer bug):** new environments must be registered in **both** `env_server.py` and `train.py`. Forgetting the second is silent and breaks training.

## The 3-terminal workflow

```
Terminal 1: Minecraft client            (cd Malmo/Minecraft && ./launchClient.bat)
Terminal 2: env server (conda: malmo)   (python Malmo/rl/envs/env_server.py --env <name> --port <P> --malmo-port 10000)
Terminal 3: training  (conda: train_env)(python Malmo/rl/training/train_sb3.py --env <name> --base-port <P>)
```

Defaults: env-server port `9999` (or `10002` for bridging examples), Malmo client port `10000`. Multi-env training launches one Minecraft client + one env_server per parallel env, all on different ports.

## Directory map

```
malmo/
  rl/
    envs/           — env classes (parkour_env.py, bridging_env.py), env_server.py, env_client.py, sb3_env_wrapper.py, per-env mission XML folders
    algos/          — base_agent.py (abstract), ppo.py, dqn.py, behavioral_cloning.py
    models/         — actor_critic.py (current multi-stream net), mlp.py (legacy)
    training/
      train.py      — main training entry (with ALGO_REGISTRY + ENV_REGISTRY)
      train_sb3.py  — StableBaselines3 variant
      configs/      — base_cfg.py + per-env <name>_cfg.py files
      curriculum.py — multi-env scheduler
    evaluation/     — evaluate.py
    utils/          — record_demos.py, logger.py
    visualization/  — trajectory and heatmap rendering
  demos/            — recorded demo JSON files
  baselines/        — shared pre-trained checkpoints
  docs/             — see pointer table below
Mod/, Schemas/, offical_install/, Python_Examples/ — upstream Malmo, do NOT modify.
```

## Documentation pointer table

When you need to extend the framework, the docs under `malmo/docs/framework/` are authoritative — read them first.

| Topic | Doc |
|---|---|
| Add a new environment / parkour variant / task type | `malmo/docs/framework/new_environment.md` |
| Add a new RL algorithm | `malmo/docs/framework/new_algorithm.md` |
| Add / swap a model architecture | `malmo/docs/framework/new_model.md` |
| What the agent perceives (per-task obs layout) | `malmo/docs/framework/observation_vector.md` |
| Available actions and how to modify them | `malmo/docs/framework/action_space.md` |
| Behavioral cloning (demos + BC→PPO pipeline) | `malmo/docs/framework/behavioral_cloning.md` |
| Curriculum / multi-env scheduling | `malmo/docs/framework/curriculum_training.md` |
| First-time setup (conda envs, Malmo install) | `malmo/docs/framework/setup.md` |
| Parkour task-specific notes | `malmo/docs/parkour/report.md` |
| Bridging task-specific notes | `malmo/docs/bridging/bridging.md` |

## Conventions

- **Names:** snake_case env name is used identically across the mission XML filename, the folder under `malmo/rl/envs/<name>/`, the config filename `<name>_cfg.py`, the config class `NameCFG`, and both `ENV_REGISTRY` keys.
- **Configs** inherit from `BaseCFG` (`malmo/rl/training/configs/base_cfg.py`) and override only what's different. Reward constants like `REWARD_SUCCESS`, `REWARD_PROGRESS_COEF`, `REWARD_STEP_PENALTY` live in the config.
- **Env classes** mirror the structure of `parkour_env.py` or `bridging_env.py` — they expose `reset()`, `step(action)`, `close()`, and return numpy obs of shape `(cfg.INPUT_SIZE,)`. The `info` dict from `step` must include `outcome`, `steps`, `pos`, `action`.
- **Algorithms** inherit from `BaseAgent` (`malmo/rl/algos/base_agent.py`) and implement `__init__`, `collect_step`, `update`, `select_action`. `save()` / `load()` come from the base class — override `_extra_state()` / `_load_extra_state()` for algo-specific state.
- **Models** implement `get_distribution(obs)`, `get_value(obs)`, `evaluate_actions(obs, actions)`. Constructor takes a single `cfg` object.
- **Reward changes invalidate checkpoints.** Flag this whenever rewards are edited.

## Sub-agent roster

Claude Code auto-dispatches to specialized sub-agents based on the user's request. Direct invocation also works (`@agent-name`).

| Agent | Use when the user wants to… |
|---|---|
| `env-builder` | add a new environment, mission, level, parkour variant, or task type |
| `reward-tuner` | tune reward constants or modify `_get_reward()` for an existing env |
| `algo-builder` | add a new RL algorithm (scaffolds a stub — math is the user's) |
| `model-swapper` | add or swap a neural network architecture |
| `framework-guide` | ask how the framework works, what the obs vector means, where something lives (read-only Q&A) |
| `experiment-orchestrator` | set up a whole experiment end-to-end from a filled-out `EXPERIMENT_TEMPLATE.md` |
| `training-runner` | launch training/eval or troubleshoot a Malmo connection / registry error |
| `change-validator` | run static + smoke checks after edits; routes failures back to the specialist that should fix them. Also fires automatically as a `PostToolUse` hook (Tier 1 only) on edits to `malmo/rl/**/*.py` or mission XML. |

## Form-driven workflow (recommended for newcomers)

If a user is new and unsure where to start, point them at `.claude/templates/EXPERIMENT_TEMPLATE.md`. They copy it to the repo root, fill in fields in plain English (name, environment description, goals, rewards, algorithm, model, launch options), and hand it to Claude with a phrase like "run experiment-orchestrator on this form." The orchestrator parses the fields and dispatches to the specialist sub-agents above in the right order.

## What NOT to touch

- `Mod/`, `Schemas/`, `offical_install/`, `Python_Examples/` — upstream Malmo assets.
- `malmo/Minecraft/` — large binary tree, gitignored.
- The 4 abstract method signatures on `BaseAgent` and the 3 method signatures on model classes. These are the contracts the training loop depends on.
