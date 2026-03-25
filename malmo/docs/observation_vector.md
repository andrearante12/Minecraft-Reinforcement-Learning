# Observation Vector

All algorithms receive the same observation — a flat float32 numpy array of 159 values built by `parkour_env.py` on every step. The observation is divided into three parts.

## Proprioception (indices 0–5)

The agent's own physical state:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `onGround` | `1.0` if standing on a block, `0.0` if airborne |
| 1 | `yaw` | Horizontal facing direction, normalized to [-1, 1] |
| 2 | `pitch` | Vertical look angle, normalized to [-1, 1] |
| 3 | `delta_y` | Vertical velocity (current Y - previous Y) |
| 4 | `delta_x` | Lateral velocity |
| 5 | `delta_z` | Forward velocity — most important for jump timing |

## Goal Delta (indices 6–8)

Vector from the agent's current position to the landing platform:

| Index | Name | Description |
|-------|------|-------------|
| 6 | `goal_dx` | `goal_x - agent_x` |
| 7 | `goal_dy` | `goal_y - agent_y` |
| 8 | `goal_dz` | `goal_z - agent_z` — how far to the target |

## Voxel Grid (indices 9–158)

A 3D snapshot of blocks surrounding the agent (5×5×6 = 150 values):

| Dimension | Range | Size |
|-----------|-------|------|
| X (lateral) | -2 to +2 | 5 |
| Y (vertical) | -1 to +3 | 5 |
| Z (forward) | -2 to +3 | 6 |

Each value is encoded as `0` (air) or `1` (solid block). The grid is asymmetric in Z — it looks 3 blocks further forward than backward since jumps always travel in the forward direction.

## Modifying the Observation

The observation is constructed in `envs/parkour_env.py` inside `_build_obs_vector()`. To change what the agent perceives — for example adding velocity history or expanding the voxel range — that is the only file that needs to be updated.

After making changes, update `INPUT_SIZE` in `training/config.py` to match the new vector length. The network is instantiated with `INPUT_SIZE` so mismatches will cause a shape error at startup rather than silently producing wrong results.
