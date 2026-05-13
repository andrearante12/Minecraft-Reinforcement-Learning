# Parkour Agent

The parkour environment teaches an RL agent to sprint-jump across gaps of increasing size and geometry in a 3D Minecraft world. The agent learns entirely from reward signals using a compact 159-dimensional observation vector — no pixels, no human demonstrations required.

> **Prerequisites**: Both conda environments set up (`malmo` + `train_env`) and Malmo installed. See [setup.md](../framework/setup.md) if not.

---

## Quick Start

Training a parkour agent takes 3 steps. You need 2 terminals running at all times (Minecraft client + env server).

### 1. Launch Minecraft + Env Server

Keep these running throughout training and evaluation.

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Environment server (malmo env, Python 3.7)
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump --port 10002 --malmo-port 10000
```

Wait for `Waiting for training script to connect...` before proceeding.

### 2. Train

```powershell
# Terminal 3
conda activate train_env

# Resume from the shared baseline checkpoint (recommended starting point):
python Malmo/rl/training/train.py --env simple_jump --algo ppo --base-port 10002 --checkpoint Malmo/rl/baselines/ppo_parkour_baseline.pt

# Train from scratch:
python Malmo/rl/training/train.py --env simple_jump --algo ppo --base-port 10002

# Resume from a saved checkpoint:
python Malmo/rl/training/train.py --env simple_jump --algo ppo --base-port 10002 --checkpoint Malmo/rl/checkpoints/ppo_simple_jump_ep500.pt
```

Checkpoints save to `Malmo/rl/checkpoints/` every 50 episodes. The baseline checkpoint in `Malmo/rl/baselines/` is the shared starting point.

### 3. Evaluate

```powershell
conda activate train_env
python Malmo/rl/evaluation/evaluate.py --env simple_jump --checkpoint Malmo/rl/checkpoints/ppo_simple_jump_ep1000.pt --episodes 10 --port 10002
```

---

## Curriculum Training

Run an adaptive curriculum that advances automatically when the agent hits a target success rate.

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: env server (the curriculum switches envs over the same connection)
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump --port 10002 --malmo-port 10000

# Terminal 3: curriculum training
conda activate train_env

# Core 3-stage curriculum: one_block_gap → simple_jump → three_block_gap
python Malmo/rl/training/train.py --curriculum Malmo/rl/training/curricula/adaptive_parkour.json --algo ppo --base-port 10002

# Full 6-stage curriculum including diagonal and vertical variants:
python Malmo/rl/training/train.py --curriculum Malmo/rl/training/curricula/full_parkour.json --algo ppo --base-port 10002
```

The curriculum switches environments automatically — no need to restart the env server.

### Curriculum Files

| File | Stages | Notes |
|------|--------|-------|
| `adaptive_parkour.json` | one_block → simple_jump → three_block_gap | Core straight-gap progression |
| `full_parkour.json` | one_block → simple_jump → diagonal_small → diagonal_medium → vertical_small → three_block_gap | All geometry types |
| `mixed_maintenance.json` | all envs simultaneously | Weighted random sampling for generalization |

---

## Multi-Environment Training

Run N Minecraft clients for faster data collection.

```powershell
# Terminal 1: Minecraft client 1 (Malmo port 10000)
cd .\Malmo\Minecraft && .\launchClient.bat
# Terminal 2: Minecraft client 2 (Malmo port 10001)
cd .\Malmo\Minecraft && .\launchClient.bat -port 10001

# Terminal 3: env server 1
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump --port 10002 --malmo-port 10000
# Terminal 4: env server 2
conda activate malmo
python Malmo/rl/envs/env_server.py --env simple_jump --port 10003 --malmo-port 10001

# Terminal 5: training with 2 envs
conda activate train_env
python Malmo/rl/training/train.py --env simple_jump --algo ppo --num-envs 2 --base-port 10002
```

---

## Environment Reference

### Environments

| Name | Gap | Notes |
|------|-----|-------|
| `one_block_gap` | 1 block | Walk-jump suffices; easiest straight gap |
| `simple_jump` | 2 blocks | Requires sprint-jump; recommended starting point |
| `three_block_gap` | 3 blocks | Precise timing; hardest straight gap |
| `diagonal_small` | 2 blocks + 1 lateral | Sprint-jump with sideways adjustment |
| `diagonal_medium` | 3 blocks + 1 lateral | Combines gap distance with diagonal offset |
| `vertical_small` | 2 blocks + height | Tests upward jumping |
| `multi_jump_course` | chained jumps | Multi-gap course (no resets between jumps) |
| `multi_jump_branch` | branching course | Multiple valid paths |

### World Layout (simple_jump)

```
Y=46 (agent level)
Y=45 (block level)

Z:  0  1  2  3  4  5  6  7
    [start      ] [gap ] [end]
    stone stone   air   stone
```

- **Start platform**: stone at z=0..3, agent spawns at z=3.5
- **Gap**: 2 blocks of air (z=4..5)
- **End platform**: stone at z=6+
- **Agent start**: x=0.5, y=46, z=3.5, yaw=180 (facing −Z/north)
- **Goal**: x=0.5, y=45, z=6.5

### Observation Space (159 dimensions)

| Indices | Size | Content |
|---------|------|---------|
| 0–5 | 6 | Proprioception: `onGround`, `yaw`, `pitch`, `delta_y`, `delta_x`, `delta_z` |
| 6–8 | 3 | Goal delta: `dx`, `dy`, `dz` to landing platform |
| 9–158 | 150 | Voxel grid 5×5×6 (air=0, stone=1) |

**Voxel grid geometry**: x[−2:+2] × y[−1:+3] × z[−2:+3] = 5×5×6 blocks. Asymmetric in Z: +3 forward visibility for jump planning, −2 behind for context. Covers one block below the agent (footing) through 3 blocks of headroom above.

**Velocity**: Position deltas (`pos_now − pos_prev`) are used rather than Malmo's reported velocity, which can lag actual physics by up to one tick.

### Action Space (15 discrete)

| Index | Action | Malmo Commands |
|-------|--------|----------------|
| 0 | move_forward | `move 1` |
| 1 | move_backward | `move -1` |
| 2 | strafe_left | `strafe -1` |
| 3 | strafe_right | `strafe 1` |
| 4 | sprint_forward | `sprint 1, move 1` |
| 5 | jump | `jump 1` |
| **6** | **sprint_jump** | **`sprint 1, move 1, jump 1`** |
| 7 | jump_forward | `move 1, jump 1` |
| 8 | sprint_jump_left | `sprint 1, move 1, strafe -1, jump 1` |
| 9 | sprint_jump_right | `sprint 1, move 1, strafe 1, jump 1` |
| 10 | look_down | `pitch 1` |
| 11 | look_up | `pitch -1` |
| 12 | turn_left | `turn -1` |
| 13 | turn_right | `turn 1` |
| 14 | no_op | *(nothing)* |

Each action is held for `STEP_DURATION = 0.15s` (≈3 Minecraft ticks). Action 6 (`sprint_jump`) is the critical action for most jumps — it pre-encodes the simultaneous sprint + move + jump combination that random exploration almost never discovers independently.

### Reward Shaping

| Event | Reward | Description |
|-------|--------|-------------|
| Landed on goal | +10.0 | Successfully reached the end platform |
| Fell | −5.0 (proximity-scaled) | Fell below `FALL_Y_THRESHOLD = 43.0` |
| Timeout (30 steps) | −5.0 (proximity-scaled) | Exceeded `MAX_STEPS` without success |
| Step penalty | −0.01 / step | Encourages efficiency |
| Z-progress | +0.5 × Δdist | Dense shaping toward goal |
| Near miss | +2.0 | Timed out within 1.5 blocks of goal |

**Proximity-scaled terminals**: fell/timeout penalty is multiplied by `(1 − proximity)`, so an agent that reached the edge and mistimed the jump is penalized less than one that never moved. Prevents the agent from learning that "getting close = bigger penalty."

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| MAX_STEPS | 30 | Steps per episode |
| STEP_DURATION | 0.15s | Hold time per action |
| TOTAL_EPISODES | 5000 | Default training budget |
| GAMMA | 0.99 | Discount factor |
| LR | 1e-4 | Learning rate (with linear decay) |
| N_STEPS | 512 | Rollout steps per PPO update |
| N_EPOCHS | 4 | Gradient passes per rollout |
| BATCH_SIZE | 64 | Minibatch size |
| CLIP_EPS | 0.2 | PPO clipping |
| ENTROPY_COEF | 0.02 | With decay to 0.001 |
| GAE_LAMBDA | 0.95 | Advantage estimation |
| OBS_NORM | True | Welford online normalization, clipped ±10 |
| REWARD_NORM | True | Std-only normalization |

---

## Key Files

| File | Purpose |
|------|---------|
| `Malmo/rl/envs/parkour_env.py` | Environment logic (rewards, obs, actions) |
| `Malmo/rl/envs/env_server.py` | TCP server bridging Malmo ↔ training process |
| `Malmo/rl/envs/env_client.py` | Client used by training scripts |
| `Malmo/rl/training/train.py` | Training entrypoint (PPO, DQN, BC + curriculum) |
| `Malmo/rl/training/configs/base_cfg.py` | Shared hyperparameters |
| `Malmo/rl/training/configs/simple_jump_cfg.py` | Config for simple_jump environment |
| `Malmo/rl/training/curricula/` | Curriculum JSON files |
| `Malmo/rl/evaluation/evaluate.py` | Evaluation script |
| `Malmo/rl/models/actor_critic.py` | Multi-stream actor-critic network (~50K params) |
| `Malmo/rl/algos/ppo.py` | Custom PPO with GAE, obs/reward norm, decay schedules |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ConnectionRefusedError` | Start Minecraft + env server before training/evaluation |
| `WinError 10013` on env server | Port reserved by Hyper-V — pick a different `--port`. Check reserved: `netsh interface ipv4 show excludedportrange protocol=tcp` |
| Mission ends before success | Increase `timeLimitMs` in the environment's XML mission file |
| Previous mission still running | Wait ~30s or restart Minecraft client |
| `N_STEPS must be divisible by --num-envs` | Ensure `N_STEPS` (default 512) divides evenly by `--num-envs` |
| Agent never attempts the jump | Check entropy coefficient — may need to increase `ENTROPY_COEF` early in training |
