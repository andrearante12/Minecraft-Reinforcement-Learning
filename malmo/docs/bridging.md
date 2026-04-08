# Bridging Environment

The bridging environment teaches an RL agent to place blocks underneath itself to cross an open gap between two platforms. This is fundamentally different from parkour (jumping) — it requires inventory management, sneaking to avoid falling off edges, and precise block placement.

## World Layout

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

## Observation Space (214 dimensions)

| Indices | Size | Content |
|---------|------|---------|
| 0-5 | 6 | Base proprioception: onGround, yaw, pitch, delta_y, delta_x, delta_z |
| 6 | 1 | Inventory count (normalized: blocks_remaining / 64) |
| 7 | 1 | Ray hit (1=crosshair on solid block face, 0=air/nothing) |
| 8-10 | 3 | Ray hit relative position (x, y, z offset from agent) |
| 11-13 | 3 | Goal delta (dx, dy, dz to end platform) |
| 14-213 | 200 | Voxel grid 5x5x8 (air=0, stone=1) |

The voxel grid is larger than parkour (5x5x8 vs 5x5x6) to give the agent visibility over the full bridge area as it advances.

## Action Space (14 discrete)

| Index | Action | Key Binding | Description |
|-------|--------|-------------|-------------|
| 0 | move_forward | W | Walk forward |
| 1 | move_backward | S | Walk backward |
| 2 | strafe_left | A | Strafe left |
| 3 | strafe_right | D | Strafe right |
| 4 | look_down | Down arrow | Look down |
| 5 | look_up | Up arrow | Look up |
| 6 | turn_left | Left arrow | Turn left |
| 7 | turn_right | Right arrow | Turn right |
| 8 | sneak | Shift | Sneak (hold/release) |
| 9 | place_block | Right-click | Place block at crosshair |
| 10 | sneak_forward | Shift+W | Walk forward while sneaking |
| 11 | sneak_backward | Shift+S | Walk backward while sneaking |
| 12 | sneak_place | Shift+Right-click | Place block while sneaking |
| 13 | no_op | (none) | Do nothing |

All actions follow the same hold/release pattern — commands are sent for `STEP_DURATION`, then released. Combined actions like `sneak_forward` bundle sneak + movement into a single step.

## Reward Shaping

| Event | Reward | Description |
|-------|--------|-------------|
| Block placed (valid) | +2.0 | Stone placed in the gap zone (z=3..7, y=45) |
| Block placed (wasteful) | -1.0 | Stone placed outside the bridge line |
| Z-progress | +0.5 * dz | Bonus for advancing to new Z positions |
| Reached end platform | +10.0 | Successfully crossed the bridge |
| Fell | -5.0 | Fell below y=43 (proximity-scaled) |
| Step penalty | -0.02 | Per-step cost to encourage efficiency |
| Timeout | -5.0 | Exceeded 150 steps |
| Near miss | +2.0 | Timed out but close to the end platform |

Block placement is detected by comparing the voxel grid between consecutive steps — when a new solid block appears where air was, a placement is registered.

## Training Pipeline

### Phase 1: Record Demonstrations

```powershell
# Terminal 1: Minecraft client
cd .\Malmo\Minecraft && .\launchClient.bat

# Terminal 2: Env server
conda activate malmo
python Malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000

# Terminal 3: Record demos
conda activate train_env
python Malmo/rl/utils/record_demos.py --env bridging --port 10002
```

Record 50+ successful bridging demonstrations. The typical bridging sequence is:
1. Toggle sneak on (Shift)
2. Walk forward to edge (Shift+W)
3. Look down at edge (Down arrow)
4. Place block (R)
5. Walk forward onto placed block (Shift+W)
6. Repeat steps 4-5 until across
7. Walk onto end platform

### Phase 2: BC Pre-training

```powershell
conda activate train_env
python Malmo/rl/training/train.py --env bridging --algo bc --demo-path demos/bridging.json --base-port 10002
```

BC trains the policy head via cross-entropy loss on demo data. Expect `bc_loss` to drop from ~2.6 to <0.5 and accuracy to rise above 60%.

### Phase 3: PPO Fine-tuning

```powershell
conda activate train_env
python Malmo/rl/training/train.py --env bridging --algo ppo --checkpoint checkpoints/ppo_bridging_bc_ep500.pt --base-port 10002
```

PPO fine-tunes both policy and critic heads. The BC-initialized policy provides a strong starting point, dramatically reducing exploration burden.

## Key Hyperparameter Differences from Parkour

| Parameter | Parkour | Bridging | Reason |
|-----------|---------|----------|--------|
| MAX_STEPS | 30 | 150 | Bridging requires many more actions |
| ENTROPY_COEF | 0.05 | 0.1 | More exploration for sequential dependencies |
| N_STEPS | 512 | 1024 | Longer rollouts for longer episodes |
| TOTAL_EPISODES | 5000 | 10000 | Harder task needs more training |
| STEP_PENALTY | -0.01 | -0.02 | Stronger efficiency incentive |
| PROPRIOCEPTION_SIZE | 6 | 11 | Extra obs: inventory, ray-cast |

## Files

- **Environment**: `Malmo/rl/envs/bridging_env.py`
- **Config**: `Malmo/rl/training/configs/bridging_cfg.py`
- **Mission XML**: `Malmo/rl/envs/bridging/missions/bridging.xml`
