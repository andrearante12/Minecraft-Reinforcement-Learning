# Behavioral Cloning + PPO Fine-tuning

A two-phase training pipeline: (1) behavioral cloning from human demonstrations to bootstrap a reasonable policy, then (2) PPO fine-tuning to optimize beyond the demonstrations.

## Why BC?

Complex parkour sequences (multi-jump, precise timing) have huge exploration spaces. Random exploration rarely discovers the right action sequences. BC pre-training gives the agent a starting policy that already knows approximately what to do, dramatically reducing the RL exploration burden.

## Pipeline Overview

```
Phase 1: Record demos          Phase 2: BC pre-train         Phase 3: PPO fine-tune
─────────────────────          ──────────────────────         ──────────────────────
Human plays through env  →     Train ActorCritic on     →    Load BC checkpoint,
recording (obs, action)        demo (obs, action) pairs      continue with PPO
                               via cross-entropy loss
                                                              --checkpoint bc.pt
                                                              --algo ppo
```

## Step 1: Record Demonstrations

Start the Minecraft client and env server as usual, then run the recorder:

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: env server
conda activate malmo
python Malmo/rl/envs/env_server.py --env one_block_gap --port 10002 --malmo-port 10000

# Terminal 3: recorder
conda activate train_env
python Malmo/rl/utils/record_demos.py --env one_block_gap --port 10002
```

### Keyboard Mapping (Standard Minecraft Controls)

Uses the same keys as Minecraft. Held key combinations are translated into our 15-action discrete space — the most specific combo wins (e.g. Ctrl+W+Space beats W alone).

| Keys held | Action | Index |
|-----------|--------|-------|
| W | move_forward | 0 |
| S | move_backward | 1 |
| A | strafe_left | 2 |
| D | strafe_right | 3 |
| Ctrl+W | sprint_forward | 4 |
| Space | jump | 5 |
| Ctrl+W+Space | sprint_jump | 6 |
| W+Space | jump_forward | 7 |
| Ctrl+W+A+Space | sprint_jump_left | 8 |
| Ctrl+W+D+Space | sprint_jump_right | 9 |
| Down arrow | look_down | 10 |
| Up arrow | look_up | 11 |
| Left arrow | turn_left | 12 |
| Right arrow | turn_right | 13 |
| (none) | no_op | 14 |
| Enter | finish episode (reset) | - |
| Esc | save and quit | - |

The translation logic lives in `translate_keys_to_action()` in `utils/record_demos.py`.

### Demo File Format

Demos are saved as JSON in `demos/`:

```json
{
  "env": "one_block_gap",
  "episodes": [
    {
      "outcome": "landed",
      "steps": [
        {"obs": [0.0, 0.5, ...], "action": 4},
        {"obs": [1.0, 0.3, ...], "action": 6}
      ]
    }
  ]
}
```

The recorder appends to existing files, so you can record across multiple sessions.

### Clearing Demos

To start fresh and delete all recorded demos for an environment:

```powershell
rm demos/bridging.json
```

### Replaying Demos

To watch recorded demos play back in Minecraft (useful for verifying demo quality before training):

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: env server
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000

# Terminal 3: replay
conda activate train_env
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002

# Replay a single episode (0-indexed):
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002 --episode 0

# Half-speed for closer inspection:
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002 --speed 0.5
```

The replay script prints each step's action name and reward, warns if the replay outcome differs from the recorded one, and pauses between episodes waiting for Enter.

### Tips for Good Demos

- Record 20+ successful episodes for reliable BC training
- Focus on episodes with clean, intentional actions
- Only successful episodes ("landed") will teach the right behavior — the BC agent learns from all episodes, so try to keep your success rate high
- You can manually edit the JSON to remove bad episodes

## Step 2: BC Pre-training

```powershell
# Same Minecraft client + env server setup as training
conda activate train_env
python Malmo/rl/training/train.py --env one_block_gap --algo bc --demo-path demos/one_block_gap.json --base-port 10002
```

BC trains the policy head on demo data via cross-entropy loss. During training, the agent also plays in the env so you get episode metrics (success rate, reward) for free via the standard logger.

### What to Expect

- `bc_loss` should decrease over updates (from ~2.7 for 15 actions → <0.5)
- `accuracy` should increase (from ~7% random → 60-90%+)
- Episode success rate should climb as the policy improves
- Training is much faster than RL since updates use offline data

### BC Hyperparameters

In `training/configs/base_cfg.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BC_EPOCHS` | 10 | Passes over demo data per update |
| `BC_BATCH_SIZE` | 64 | Minibatch size for BC updates |
| `DEMO_PATH` | None | Set via `--demo-path` CLI arg |

## Step 3: PPO Fine-tuning

Load the BC checkpoint and switch to PPO:

```powershell
conda activate train_env
python Malmo/rl/training/train.py --env one_block_gap --algo ppo --checkpoint checkpoints/ppo_one_block_gap_bc_ep500.pt --base-port 10002
```

### How the Transition Works

1. BC saves checkpoints in the same format as PPO (via `BaseAgent.save()`)
2. PPO loads the model weights, getting a warm-started policy
3. PPO's critic head starts from BC's untrained weights — this is fine, GAE bootstraps value estimates within a few updates
4. PPO creates its own optimizer (the BC optimizer state is ignored)

### Expected Behavior

- Initial PPO episodes should already show reasonable behavior (from BC policy)
- Value loss will be high initially as the critic learns from scratch
- Policy should improve beyond BC performance within a few hundred episodes
- Entropy may spike briefly as PPO adds exploration noise to the BC policy

## Architecture Details

### BC Agent (`algos/behavioral_cloning.py`)

- Inherits from `BaseAgent` — same interface as PPO/DQN
- Uses the same `ActorCritic` model with no modifications
- `collect_steps()` plays the current policy in envs (for evaluation/logging)
- `update()` trains exclusively on demo data (cross-entropy on policy logits)
- `buffer_full()` triggers updates every `N_STEPS` collection steps
- Only the actor head gets gradient; critic head is untouched

### Demo Recorder (`utils/record_demos.py`)

- Connects to env server via `EnvClient` (same TCP protocol as training)
- Uses the `keyboard` package for real-time key detection
- Records at 20Hz by default (configurable via `--tick-rate`)
- Auto-saves after each episode
- Appends to existing demo files
