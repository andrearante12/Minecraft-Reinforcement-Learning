# Action Space

The agent operates in a **discrete action space** — on each step it selects one action from a fixed list defined in the environment config. Actions are defined as tuples of `(name, commands_on, commands_off)` in `training/configs/<env>_cfg.py`.

Each action sends one or more Malmo commands when it starts, holds them for `STEP_DURATION` seconds, then sends the corresponding release commands. This means actions are temporally extended — the agent commits to a movement for a fixed duration rather than issuing instantaneous commands.

---

## Current Action Set

| Index | Name | Commands | Description |
|-------|------|----------|-------------|
| 0 | `move_forward` | `move 1` | Walk forward |
| 1 | `move_backward` | `move -1` | Walk backward |
| 2 | `strafe_left` | `strafe -1` | Sidestep left without turning |
| 3 | `strafe_right` | `strafe 1` | Sidestep right without turning |
| 4 | `sprint_forward` | `sprint 1` + `move 1` | Sprint forward — covers more distance per step |
| 5 | `jump` | `jump 1` | Jump in place |
| 6 | `sprint_jump` | `sprint 1` + `move 1` + `jump 1` | Sprint and jump simultaneously — the primary jump for crossing gaps |
| 7 | `look_down` | `pitch 1` | Tilt camera downward |
| 8 | `look_up` | `pitch -1` | Tilt camera upward |
| 9 | `turn_left` | `turn -1` | Rotate left |
| 10 | `turn_right` | `turn 1` | Rotate right |
| 11 | `no_op` | — | Do nothing — allows the agent to wait or hold position |

---

## Malmo Command Reference

Actions are built from Malmo's `ContinuousMovementCommands`. All values are continuous floats clamped to [-1, 1]:

| Command | Range | Effect |
|---------|-------|--------|
| `move <v>` | -1 to 1 | Forward/backward movement. `1` = full forward, `-1` = full backward |
| `strafe <v>` | -1 to 1 | Lateral movement. `1` = right, `-1` = left |
| `turn <v>` | -1 to 1 | Yaw rotation. `1` = right, `-1` = left |
| `pitch <v>` | -1 to 1 | Vertical look angle. `1` = down, `-1` = up |
| `jump <v>` | 0 or 1 | Jump. `1` = jumping, `0` = stop jumping |
| `sprint <v>` | 0 or 1 | Sprint modifier. `1` = sprinting, `0` = normal speed |

Commands remain active until a `0` (or release command) is sent. `_take_action()` in `env.py` handles this — it sends all `commands_on` at the start of a step, waits `STEP_DURATION` seconds, then sends all `commands_off`.

---

## Step Duration

`STEP_DURATION` in the config controls how long each action is held. The default is `0.15` seconds.

- **Shorter** (e.g. `0.1s`) — finer-grained control, more steps per episode, harder to learn timing
- **Longer** (e.g. `0.3s`) — coarser control, fewer steps, easier to learn basic movements but harder to time jumps precisely

---

## Adding or Removing Actions

Actions are defined per-environment in the config. To add a new action, append a tuple to the `ACTIONS` list and update `N_ACTIONS`:

```python
ACTIONS = [
    ...
    ("sprint_jump_left", ["sprint 1", "move 1", "strafe -1", "jump 1"],
                         ["sprint 0", "move 0", "strafe 0",  "jump 0"]),
]
N_ACTIONS = len(ACTIONS)
```

To remove an action, delete its entry from the list. `N_ACTIONS` is computed automatically via `len(ACTIONS)`.

If you change `N_ACTIONS`, the policy head output size changes — any saved checkpoints trained with the old action count will be incompatible and cannot be loaded.

---
