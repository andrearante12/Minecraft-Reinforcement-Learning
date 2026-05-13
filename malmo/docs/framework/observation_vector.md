# Observation Vector

### Proprioception (indices 0–5)

  | Index | Name | Description |
  |-------|------|-------------|
  | 0 | `onGround` | `1.0` if standing on a block, `0.0` if airborne |
  | 1 | `yaw` | Horizontal facing direction, normalized to [-1, 1] |
  | 2 | `pitch` | Vertical look angle, normalized to [-1, 1] |
  | 3 | `delta_y` | Vertical velocity (current Y - previous Y) |
  | 4 | `delta_x` | Lateral velocity |
  | 5 | `delta_z` | Forward velocity — most important for jump timing |

  ### Goal Delta (indices 6–8)

  | Index | Name | Description |
  |-------|------|-------------|
  | 6 | `goal_dx` | `goal_x - agent_x` |
  | 7 | `goal_dy` | `goal_y - agent_y` |
  | 8 | `goal_dz` | `goal_z - agent_z` |

  ### Voxel Grid (indices 9–158) — 5×5×6 = 150 values

  | Dimension | Range | Size |
  |-----------|-------|------|
  | X (lateral) | -2 to +2 | 5 |
  | Y (vertical) | -1 to +3 | 5 |
  | Z (forward) | -2 to +3 | 6 |

  Each value is `0` (air) or `1` (solid block). The grid is asymmetric in Z
  — it looks further forward than backward since jumps always travel in the
  forward direction.

  ---

  ## Bridging Environment (`bridging_env.py`) — 220 values

  The bridging environment extends the base observation with inventory and
  ray-cast data needed for block placement, and uses a larger voxel grid.

  ### Proprioception — base (indices 0–8)

  | Index | Name | Description |
  |-------|------|-------------|
  | 0 | `onGround` | `1.0` if standing on a block, `0.0` if airborne |
  | 1 | `yaw` | Horizontal facing direction, normalized to [-1, 1] |
  | 2 | `pitch` | Vertical look angle, normalized to [-1, 1] |
  | 3 | `vel_y` | Vertical velocity (frame delta) |
  | 4 | `vel_x` | Lateral velocity (frame delta) |
  | 5 | `vel_z` | Forward velocity (frame delta) |
  | 6 | `pos_x` | X position relative to spawn |
  | 7 | `pos_y` | Y position relative to spawn |
  | 8 | `pos_z` | Z position relative to spawn |

  ### Proprioception — bridging-specific (indices 9–16)

  | Index | Name | Description |
  |-------|------|-------------|
  | 9  | `inv_count` | Blocks remaining in inventory, normalized to [0, 1] |
  | 10 | `ray_hit` | `1.0` if crosshair targets a solid block face, else `0.0` |
  | 11 | `ray_rel_x` | X offset of targeted block from agent |
  | 12 | `ray_rel_y` | Y offset of targeted block from agent |
  | 13 | `ray_rel_z` | Z offset of targeted block from agent |
  | 14 | `face_v` | Targeted face vertical axis: `+1` top, `-1` bottom, `0` side |
  | 15 | `face_dx` | Targeted face east/west: `+1` east, `-1` west, `0` other |
  | 16 | `face_dz` | Targeted face south/north: `+1` south, `-1` north, `0` other |

  The ray-cast features (10–16) are critical for block placement — the agent
   must be looking at the south face of the last placed block to extend the
  bridge forward.

  ### Goal Delta (indices 17–19)

  | Index | Name | Description |
  |-------|------|-------------|
  | 17 | `goal_dx` | `goal_x - agent_x` |
  | 18 | `goal_dy` | `goal_y - agent_y` |
  | 19 | `goal_dz` | `goal_z - agent_z` |

  ### Voxel Grid (indices 20–219) — 5×5×8 = 200 values

  | Dimension | Range | Size |
  |-----------|-------|------|
  | X (lateral) | -2 to +2 | 5 |
  | Y (vertical) | -2 to +2 | 5 |
  | Z (forward) | -2 to +5 | 8 |

  Each value is `0` (air) or `1` (solid block). The grid extends further
  forward than the parkour grid to give the agent visibility over the full
  gap ahead.

  ---

  ## Modifying the Observation

  Each environment builds its observation in its own `_build_obs_vector()`
  method. To add or remove features, edit that method and update
  `INPUT_SIZE` in the corresponding config file
  (`training/configs/<env>_cfg.py`). The network is instantiated with
  `INPUT_SIZE` so a mismatch will cause a shape error at startup rather than
   silently producing wrong results.
