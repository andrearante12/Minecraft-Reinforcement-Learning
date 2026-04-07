# Visualization — Replay & Heatmap Tool

Interactive 3D viewer for post-training analysis of agent trajectories. Supports **Replay** (per-episode step-through with playback) with a **Heatmap** (aggregate visit density across episodes) feature.

---

## Quick Start

```powershell
conda activate train_env

# Replay mode (default)
python malmo/rl/visualization/replay.py --run malmo/rl/logs/<run_prefix>

# Use sample data to test without a training run
python malmo/rl/visualization/replay.py --run malmo/rl/logs/sample_simple_jump_ppo_20260401_143022
```

If `--run` points to a directory containing multiple runs, a numbered selection prompt appears in the terminal.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--run` | *(required)* | Path to a run prefix, a `_trajectories.csv` file, or a directory of runs |
| `--episode N` | first ep | Episode to start the replay at |
| `--episodes 1,5,10-20` | all | Episode filter for heatmap mode |
| `--outcome landed` | all | Outcome filter for heatmap mode (`landed`, `fell`, `timeout`, `near_miss`) |
| `--colormap hot` | `hot` | Matplotlib colormap for heatmap scatter |
| `--theme dark\|light` | `dark` | Colour theme |
| `--interval N` | `150` | Playback speed in ms per step (150 ms matches real-time STEP_DURATION) |

---

## Replay Mode

The replay window shows the 3D world geometry with block outlines, spawn marker (green star), and goal marker (gold diamond). The current episode's trajectory is drawn as a coloured line and an animated dot marks the agent's current position.

### Trajectory colours

| Outcome | Colour |
|---|---|
| `landed` | Green |
| `fell` | Red |
| `timeout` | Grey |
| `near_miss` | Yellow |
| `alive` (mid-ep) | Blue |

### Bottom controls

**Row 1 — Heatmap up to ep slider** (`0.08–0.68` width)
Controls which episodes contribute to the heatmap overlay. Defaults to the last episode (all episodes included). Only active when *Toggle Heatmap* is checked.

**Row 2 — Episode slider**
Scrub to any episode. Dragging stops playback.

**Row 3 — Step slider**
Scrub to any step within the current episode.

**Play/Pause button** (right of step slider)
Starts/stops animated playback at the rate set by `--interval`.

**Checkboxes** (far right, top to bottom):
- **Repeat ep** *(default: on)* — at the end of an episode, restart from step 0 instead of advancing to the next episode.
- **Toggle Heatmap** *(default: off)* — overlay the accumulated visit-density heatmap on the 3D world. Activates the *Heatmap up to ep* slider and *Show Final Heatmap Only* toggle.
- **Show Final Heatmap Only** *(default: off)* — when heatmap is on, ignore the *Heatmap up to ep* slider and show the full accumulated heatmap of all episodes. Toggle off to restore the slider-controlled accumulation.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `→` / `←` | Next / previous step |
| `↑` / `↓` | Next / previous episode |
| `Home` | Jump to first step |
| `End` | Jump to last step |
| `r` | Reset 3D camera to default angle |

### Info panel (right side)

Displays run name, environment, current episode number and outcome, step-level position/action/reward data, and an outcome breakdown across all episodes.

### Heatmap overlay behaviour

When *Toggle Heatmap* is checked, a scatter plot is drawn on the world at the standing Y-level (the most-visited Y layer). Point size and colour intensity reflect relative visit frequency. The overlay is redrawn whenever:
- The *Heatmap up to ep* slider changes (and *Show Final Heatmap Only* is off)
- *Toggle Heatmap* or *Show Final Heatmap Only* is toggled

The overlay does **not** clear when changing episodes — it reflects all episodes up to the slider value, independent of which episode's trajectory is displayed.

---

## Heatmap Mode (`--heatmap`)

A standalone window showing aggregate visit density across episodes. No trajectory is shown — only world geometry and the density scatter.

### Bottom controls

**Episode accumulation slider** — defaults to the last episode (all episodes). Drag left to see density from only the first N episodes, allowing you to watch the agent's behaviour evolve over training.

**Show Final Heatmap Only** checkbox — when checked, ignores the slider and displays the full dataset. When unchecked, the slider is active again.

The heatmap is re-rendered on every slider change and toggle click. The title updates to show how many episodes are included.

### Expected appearance

- **Spawn area**: brightest/hottest — the agent always starts here.
- **Gap region**: sparse — mid-air time is brief.
- **Goal platform**: density grows as the agent learns to land; absent in early training.
- With `--outcome landed`: only successful episodes shown; goal platform dominates.

---

## Data Sources

The tool reads `_trajectories.csv` files for full per-step data. If only `_episodes.csv` exists (no trajectory file), world geometry is rendered from the mission XML but no trajectory or heatmap data is available.

The `logs/` directory contains a `_gen_sample_logs.py` script that generates realistic synthetic data for testing without a live training run.
