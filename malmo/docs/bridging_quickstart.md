# Bridging Agent — Quick Start

Get a bridging agent trained from scratch in 4 steps: record demos, BC pre-train, PPO fine-tune, evaluate.

> **Prerequisites**: Both conda environments set up (`malmo` + `train_env`) and Malmo installed. See [setup.md](setup.md) if not.

---

## 1. Launch Minecraft + Env Server

You always need these two terminals running before anything else.

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Environment server (malmo env, Python 3.7)
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
```

Wait for `Waiting for training script to connect...` before proceeding.

---

## 2. Record Human Demonstrations

Record 50+ successful bridging episodes. These are used for behavioral cloning.

```powershell
# Terminal 3: Demo recorder (train_env, Python 3.10)
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
1. Press Shift (sneak_down) — prevents falling off edges while building
2. Walk forward to the gap edge (W)
3. Look down (Down arrow) so the crosshair targets the block face below
4. Right-click to place a block
5. Walk forward onto the placed block (W)
6. Repeat until you reach the end platform
7. Release Shift (sneak_up) and walk onto the platform

> Block placement is detected via inventory count — even if a right-click isn't captured by polling, the recorder auto-corrects using the env's inventory tracking. You'll see `[PLACED #N]` in the logs.

Demos save to `Malmo/demos/bridging.json`. To start fresh: `del Malmo\demos\bridging.json`

---

## 3. Train the Agent

Training uses Stable Baselines3 PPO. Install once:

```powershell
conda activate train_env
pip install stable-baselines3
```

### Option A: BC pre-training + PPO (recommended)

```powershell
# BC pre-trains on demos, then switches to PPO
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --demo-path demos/bridging.json
```

### Option B: PPO from scratch

```powershell
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002
```

### Resume from checkpoint

```powershell
python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --checkpoint checkpoints/sb3_bridging_100000_steps.zip
```

### Multi-environment training

For faster training, run multiple Minecraft instances:

```powershell
# Terminal 1: Minecraft client 1 (port 10000)
cd .\Malmo\Minecraft && .\launchClient.bat
# Terminal 2: Minecraft client 2 (port 10001)
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

## 4. Evaluate

```powershell
# Env server must be running (step 1)
conda activate train_env
python Malmo/rl/evaluation/evaluate.py --env bridging --checkpoint checkpoints/sb3_bridging_final.zip --episodes 50 --port 10002
```

---

## World Layout

```
Y=46 (agent)    [start   ] [   5-block gap   ] [  end    ]
Y=45 (blocks)    stone      air air air air air  stone
Z:               0  1  2    3  4  5  6  7        8  9  10
```

- Start: z=0..2, End: z=8..10, Gap: z=3..7
- Agent spawns at (0.5, 46, 1.5) looking down (pitch=70)
- 64 stone blocks in inventory

## Action Space (12 discrete)

| Idx | Action | Idx | Action |
|-----|--------|-----|--------|
| 0 | move_forward | 6 | turn_left |
| 1 | move_backward | 7 | turn_right |
| 2 | strafe_left | 8 | sneak_down |
| 3 | strafe_right | 9 | sneak_up |
| 4 | look_down | 10 | place_block |
| 5 | look_up | 11 | no_op |

## Reward Structure

| Event | Reward |
|-------|--------|
| Block placed | +2.0 per block |
| Z-progress | +0.5 per new z reached |
| Reached end | +10.0 |
| Fell (y < 43) | -5.0 (proximity-scaled) |
| Step penalty | -0.02 per step |
| Timeout (300 steps) | -5.0 (proximity-scaled) |
| Near miss | +2.0 |

## Key Files

| File | Purpose |
|------|---------|
| `Malmo/rl/envs/bridging_env.py` | Environment logic (rewards, obs, actions) |
| `Malmo/rl/envs/bridging/missions/bridging.xml` | Minecraft world definition |
| `Malmo/rl/training/configs/bridging_cfg.py` | Hyperparameters and reward values |
| `Malmo/rl/training/train_sb3.py` | SB3 PPO training script |
| `Malmo/rl/envs/sb3_env_wrapper.py` | Gymnasium wrapper for SB3 |
| `Malmo/rl/utils/record_demos.py` | Demo recorder |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ConnectionRefusedError` | Start Minecraft + env server before training/recording |
| `mission_ended` before success | Increase `timeLimitMs` in bridging.xml (currently 120000ms) |
| SB3 shows `ep_rew_mean = 0` | Ensure `Monitor` wrapper is in `train_sb3.py` make_env() |
| Malmo port conflict (`WinError 10013`) | Pick different `--malmo-port`. Check reserved: `netsh interface ipv4 show excludedportrange protocol=tcp` |
| Previous mission still running | Wait ~30s or restart Minecraft client |
