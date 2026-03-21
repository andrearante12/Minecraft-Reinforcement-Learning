# Curriculum Training

Train an agent across multiple environments in a single run. All environments in a curriculum must share the same `INPUT_SIZE` and `N_ACTIONS`.

## JSON Format

### Sequential Mode

Fixed stages, each with a set number of episodes:

```json
{
  "mode": "sequential",
  "stages": [
    {"env": "simple_jump", "episodes": 500},
    {"env": "one_block_gap", "episodes": 500},
    {"env": "three_block_gap", "episodes": 1000}
  ]
}
```

Total episodes = sum of all stage episodes (2000 in this example).

### Weighted Mode

Random sampling with weights per environment:

```json
{
  "mode": "weighted",
  "total_episodes": 2000,
  "seed": 42,
  "envs": [
    {"env": "simple_jump", "weight": 0.6},
    {"env": "three_block_gap", "weight": 0.4}
  ]
}
```

Env selection is deterministic given the seed + episode number, so checkpoint resume reproduces the same schedule.

### Adaptive Mode

Performance-gated progression — advance to the next stage only when the agent hits a target success rate:

```json
{
  "mode": "adaptive",
  "stages": [
    {"env": "one_block_gap",   "target_success_rate": 0.9, "window": 50, "min_episodes": 100, "max_episodes": 2000},
    {"env": "simple_jump",     "target_success_rate": 0.9, "window": 50, "min_episodes": 100, "max_episodes": 2000},
    {"env": "three_block_gap", "max_episodes": 3000}
  ]
}
```

**Stage fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `env` | (required) | Environment name |
| `target_success_rate` | `null` | Success rate threshold to advance (not needed on last stage) |
| `window` | `50` | Rolling window size for computing success rate |
| `min_episodes` | `100` | Minimum episodes before promotion is allowed |
| `max_episodes` | `2000` | Safety cap — advance even if target not met |

**How promotion works:**

1. After each episode, the outcome ("landed" = success) is added to a rolling window.
2. Promotion happens when **both**: (a) `min_episodes` have been completed on the current stage, AND (b) the rolling success rate over the last `window` episodes meets `target_success_rate`, OR `max_episodes` is reached.
3. The last stage has no target — it runs until `max_episodes`.
4. A banner is printed on each promotion showing the achieved success rate.

Total episodes = sum of all `max_episodes` (upper bound; training may finish earlier if all stages promote quickly).

**Checkpoint resume:** Adaptive mode saves a sidecar file (`*_curriculum.pt`) alongside the agent checkpoint. This stores which stage the agent is on, the episode count, and the outcome history. On resume, the scheduler restores its full state.

## Usage

```powershell
# 1. Start Minecraft clients and env servers as usual (see CLAUDE.md)
# 2. Run training with --curriculum instead of --env:
conda activate train_env
python Malmo/rl/training/train.py --curriculum path/to/curriculum.json --algo ppo --base-port 10002
```

The `--env` flag is still used for single-environment training (no curriculum file needed).

## Checkpoint Resume

Episode number is parsed from the checkpoint filename. The scheduler deterministically computes which env to use for that episode, so training resumes seamlessly:

```powershell
python Malmo/rl/training/train.py --curriculum path/to/curriculum.json --algo ppo \
    --checkpoint checkpoints/ppo_curriculum_ppo_ep500.pt
```

## Multi-Env + Curriculum

With `--num-envs 2`, each env slot independently queries the scheduler. Different slots may be on different curriculum stages if they finish episodes at different rates.

## How It Works

- `CurriculumScheduler` (`training/curriculum.py`) validates env compatibility and provides `env_for_episode(n)`.
- `env_server.py` supports a `switch_env` protocol command that swaps the running environment without restarting the server.
- `env_client.py` exposes `switch_env(env_name)` for the training script to call.
- The training loop checks at each episode boundary whether the next episode needs a different env. Switching only happens when needed (no overhead for consecutive same-env episodes).

## Validation

The scheduler validates at startup that:
1. All referenced env names exist in the registry.
2. All envs share the same `INPUT_SIZE` and `N_ACTIONS`.

Mismatches raise a `ValueError` before training begins.

## Logging

Episode CSVs include an `env` column showing which environment produced each episode. Console summaries show per-env breakdowns when multiple environments are detected.
