# Bridging Agent

> **Linux note:** commands below use Windows syntax. On Linux, use `cd malmo/Minecraft && ./launchClient.sh` instead of `cd .\Malmo\Minecraft && .\launchClient.bat`, and lowercase `malmo/rl/...` paths. See [setup_linux.md](../framework/setup_linux.md).

The bridging environment teaches an RL agent to place blocks underneath itself to cross an open gap between two platforms. Unlike parkour (jumping), bridging requires inventory management, sneaking to avoid falling, and precise block placement.

> **Prerequisites**: Both conda environments set up (`malmo` + `train_env`) and Malmo installed. See [setup.md](../framework/setup.md) if not.

---

## Quick Start

Training a bridging agent takes 4 steps. You need 2 terminals running at all times (Minecraft client + env server).

### 1. Launch Minecraft + Env Server

Keep these running throughout recording, training, and evaluation.

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Environment server (malmo env, Python 3.7)
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
```

Wait for `Waiting for training script to connect...` before proceeding.

### 2. Record Human Demonstrations

Record 50+ successful bridging episodes for behavioral cloning.

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
| Shift (press) | sneak_down — begin crouching |
| Shift (release) | sneak_up — stop crouching |
| Right-click | Place block |
| Arrow keys | Look / turn |
| Esc | Save & quit |

**Bridging technique:**
1. Press Shift (sneak_down) — prevents falling while building
2. Walk forward to the gap edge (W)
3. Look down (Down arrow) so the crosshair targets the block face below
4. Right-click to place a block
5. Walk forward onto the placed block (W)
6. Repeat until you reach the end platform
7. Release Shift (sneak_up) and walk onto the platform

Demos save to `Malmo/demos/bridging.json`. To start fresh: `del Malmo\demos\bridging.json`

Replay demos to verify quality (env server must be running):

```powershell
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002
python Malmo/rl/utils/replay_demos.py --env bridging --port 10002 --episode 0 --speed 0.5
```

### 3. Train

```powershell
# Resume from the shared baseline checkpoint (recommended starting point):
conda activate train_env
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --checkpoint Malmo/rl/baselines/sb3_bridging_baseline.zip

# BC pre-training on demos, then PPO fine-tuning:
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --demo-path Malmo/demos/bridging.json

# PPO from scratch (no demos):
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002

# Resume from any saved checkpoint:
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --checkpoint Malmo/rl/checkpoints/sb3_bridging_<N>_steps.zip
```

Checkpoints save to `Malmo/rl/checkpoints/`. The baseline checkpoint in `Malmo/rl/baselines/` is the shared starting point.

### 4. Evaluate

```powershell
conda activate train_env
# Diagonal bridging (default — GOAL_X_OFFSETS=[4]):
python Malmo/rl/evaluation/evaluate_bridging.py --env bridging --checkpoint Malmo/rl/checkpoints/sb3_bridging_final.zip --episodes 10 --port 10002

# Straight bridging (GOAL_X_OFFSETS=[0]):
python Malmo/rl/evaluation/evaluate_bridging.py --env bridging_straight --checkpoint Malmo/rl/checkpoints/sb3_bridging_final.zip --episodes 10 --port 10002
```

Note: restart the env server with `--env bridging_straight` when switching variants.

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

---

## Environment Reference

### World Layout

```
Y=46 (agent level)
Y=45 (block level)

Z:  0  1  2  3  4  5  6  7  8  9  10
    [start   ] [   5-block gap   ] [  end    ]
    stone      air air air air air  stone
```

- **Start platform**: 3-wide stone at z=0..2, x=-1..1
- **End platform**: 3-wide stone at z=8..10, x=-1..1
- **Gap**: 5 blocks of air (z=3..7) at y=45
- **Side walls**: Stone walls at x=-2 and x=2 to prevent wandering
- **Agent start**: x=0.5, y=46, z=1.5, looking down (pitch=70)
- **Inventory**: 64 stone blocks in hotbar slot 0

### Observation Space (214 dimensions)

| Indices | Size | Content |
|---------|------|---------|
| 0–5 | 6 | Base proprioception: onGround, yaw, pitch, delta_y, delta_x, delta_z |
| 6 | 1 | Inventory count (normalized: blocks_remaining / 64) |
| 7 | 1 | Ray hit (1=crosshair on solid block face, 0=air/nothing) |
| 8–10 | 3 | Ray hit relative position (x, y, z offset from agent) |
| 11–13 | 3 | Goal delta (dx, dy, dz to end platform) |
| 14–213 | 200 | Voxel grid 5×5×8 (air=0, stone=1) |

The voxel grid is larger than parkour (5×5×8 vs 5×5×6) to give the agent visibility over the full bridge area as it advances.

### Action Space (12 discrete)

| Index | Action | Description |
|-------|--------|-------------|
| 0 | move_forward | Walk forward |
| 1 | move_backward | Walk backward |
| 2 | strafe_left | Strafe left |
| 3 | strafe_right | Strafe right |
| 4 | look_down | Look down |
| 5 | look_up | Look up |
| 6 | turn_left | Turn left |
| 7 | turn_right | Turn right |
| 8 | sneak_down | Begin crouching (edge-triggered, persistent until sneak_up) |
| 9 | sneak_up | Stop crouching (edge-triggered) |
| 10 | place_block | Place block at crosshair |
| 11 | no_op | Do nothing |

Sneak is edge-triggered: `sneak_down` fires once on press and `sneak_up` fires once on release. Any movement action issued while sneaking re-applies `crouch 1` after it completes to preserve persistent state.

### Reward Shaping

| Event | Reward | Description |
|-------|--------|-------------|
| Block placed (valid) | +2.0 | Stone placed in the gap zone (z=3..7, y=45) |
| Block placed (wasteful) | -1.0 | Stone placed outside the bridge line |
| Z-progress | +0.5 × dz | Bonus for advancing to new Z positions |
| Reached end platform | +10.0 | Successfully crossed the bridge |
| Fell | -5.0 | Fell below y=43 (proximity-scaled) |
| Step penalty | -0.02 | Per-step cost to encourage efficiency |
| Timeout | -5.0 | Exceeded 150 steps |
| Near miss | +2.0 | Timed out but close to the end platform |

Block placement is detected by comparing the voxel grid between consecutive steps — when a new solid block appears where air was, a placement is registered.

### Hyperparameter Differences from Parkour

| Parameter | Parkour | Bridging | Reason |
|-----------|---------|----------|--------|
| MAX_STEPS | 30 | 150 | Bridging requires many more actions |
| ENTROPY_COEF | 0.05 | 0.1 | More exploration for sequential dependencies |
| N_STEPS | 512 | 1024 | Longer rollouts for longer episodes |
| TOTAL_EPISODES | 5000 | 10000 | Harder task needs more training |
| STEP_PENALTY | -0.01 | -0.02 | Stronger efficiency incentive |
| PROPRIOCEPTION_SIZE | 6 | 11 | Extra obs: inventory, ray-cast |

---

## Key Files

| File | Purpose |
|------|---------|
| `Malmo/rl/envs/bridging_env.py` | Environment logic (rewards, obs, actions) |
| `Malmo/rl/envs/bridging/missions/bridging.xml` | Minecraft world definition |
| `Malmo/rl/training/configs/bridging_cfg.py` | Hyperparameters and reward values |
| `Malmo/rl/training/train_sb3.py` | SB3 PPO training script |
| `Malmo/rl/envs/sb3_env_wrapper.py` | Gymnasium wrapper for SB3 |
| `Malmo/rl/utils/record_demos.py` | Demo recorder |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ConnectionRefusedError` | Start Minecraft + env server before training/recording |
| `WinError 10013` on env server | Port reserved by Hyper-V — pick a different `--port`. Check reserved: `netsh interface ipv4 show excludedportrange protocol=tcp` |
| Mission ends before success | Increase `timeLimitMs` in `bridging.xml` (currently 120000ms) |
| Previous mission still running | Wait ~30s or restart Minecraft client |
| SB3 shows `ep_rew_mean = 0` | Ensure `Monitor` wrapper is in `train_sb3.py` make_env() |
