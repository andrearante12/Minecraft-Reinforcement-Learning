"""
visualization/replay.py
------------------------
CLI entry point for the post-run replay and heatmap visualization tool.

Usage
-----
# Replay a specific run (trajectory CSV)
python malmo/rl/visualization/replay.py --run logs/simple_jump_ppo_20260401_143022

# Browse all runs in a directory
python malmo/rl/visualization/replay.py --run logs/

# Heatmap of "landed" episodes only
python malmo/rl/visualization/replay.py --run logs/ --heatmap --outcome landed

# Start at episode 42
python malmo/rl/visualization/replay.py --run logs/ --episode 42
"""

import argparse
import os
import sys

# Ensure the rl/ root is on sys.path regardless of where the script is called from
_RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from visualization.data_loader    import discover_runs, load_run
from visualization.world_renderer import (
    render_world, configure_axes, render_special_markers,
)
from visualization.traj_renderer  import (
    OUTCOME_COLORS, render_trajectory, update_dot, clear_trajectory,
)
from visualization.heatmap_renderer import accumulate_density, render_heatmap


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

_THEMES = {
    "dark": {
        "bg":         "#1A1A2E",
        "text":       "#E0E0E0",
        "slider_ep":  "#4488FF",
        "slider_step":"#44BB88",
        "panel_bg":   "#0D0D1A",
    },
    "light": {
        "bg":         "#F5F5F5",
        "text":       "#111111",
        "slider_ep":  "#2255BB",
        "slider_step":"#228844",
        "panel_bg":   "#EAEAF5",
    },
}


def _apply_theme(fig, ax3d, info_ax, theme_name):
    t = _THEMES.get(theme_name, _THEMES["dark"])
    fig.patch.set_facecolor(t["bg"])
    ax3d.set_facecolor(t["bg"])
    info_ax.set_facecolor(t["panel_bg"])
    for spine in ax3d.spines.values():
        spine.set_edgecolor(t["text"])
    ax3d.tick_params(colors=t["text"])
    ax3d.xaxis.label.set_color(t["text"])
    ax3d.yaxis.label.set_color(t["text"])
    ax3d.zaxis.label.set_color(t["text"])
    return t


# ---------------------------------------------------------------------------
# Info panel
# ---------------------------------------------------------------------------

def _outcome_summary(episodes):
    counts = {}
    total  = len(episodes)
    for ep_data in episodes.values():
        o = ep_data.get("outcome", "alive")
        counts[o] = counts.get(o, 0) + 1
    lines = []
    for outcome in ("landed", "fell", "timeout", "near_miss", "alive"):
        if outcome in counts:
            pct = 100 * counts[outcome] / total if total else 0
            lines.append(f"  {outcome:<10} {counts[outcome]:>4}  ({pct:.0f}%)")
    return lines, total


def _update_info_panel(info_ax, state, run_data, run_name, theme_key, total_eps):
    info_ax.clear()
    info_ax.axis("off")
    t = _THEMES.get(theme_key, _THEMES["dark"])

    ep  = state["current_ep"]
    ep_data = run_data.episodes.get(ep, {})
    steps   = ep_data.get("steps", [])
    outcome = ep_data.get("outcome", "alive")
    n_steps = len(steps)
    sidx    = state.get("step_idx", 0)

    lines = [
        f"Run:  {run_name}",
        f"Env:  {run_data.env_name}",
        "",
        f"Episode  {ep} / {total_eps}",
        f"Outcome: {outcome}",
        f"Steps:   {n_steps}",
        "",
    ]

    if steps and sidx < len(steps):
        s = steps[sidx]
        # env per-step (curriculum)
        ep_env = s.get("env", run_data.env_name)
        lines += [
            f"Step {sidx}",
            f"  env:    {ep_env}",
            f"  X: {s['x']:.2f}",
            f"  Y: {s['y']:.2f}",
            f"  Z: {s['z']:.2f}",
            f"  yaw:    {s['yaw']:.1f}°",
            f"  ground: {s['on_ground']}",
            f"  action: {s['action']}",
            f"  reward: {s['reward']:+.3f}",
            "",
        ]
    else:
        lines += ["(no step data)", ""]

    summary_lines, total = _outcome_summary(run_data.episodes)
    lines.append(f"Outcomes ({total} eps):")
    lines.extend(summary_lines)

    full_text = "\n".join(lines)
    info_ax.text(
        0.05, 0.97, full_text,
        transform=info_ax.transAxes,
        va="top", ha="left",
        fontsize=7.5,
        family="monospace",
        color=t["text"],
        wrap=False,
    )


# ---------------------------------------------------------------------------
# Heatmap mode
# ---------------------------------------------------------------------------

def _run_heatmap(run_data, run_name, args):
    outcome_filter = args.outcome or None

    # Episode filtering by index list / range
    episodes = run_data.episodes
    if args.episodes:
        episodes = {
            ep: data for ep, data in episodes.items()
            if ep in args.episodes
        }

    density_grid, grid_meta = accumulate_density(
        episodes,
        outcome_filter=outcome_filter,
        resolution=0.25,
    )

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("#1A1A2E")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1A1A2E")
    ax.view_init(elev=35, azim=-55)

    title = f"Heatmap — {run_name}"
    if outcome_filter:
        title += f"  [outcome={outcome_filter}]"
    ax.set_title(title, color="#E0E0E0", fontsize=10, pad=10)

    configure_axes(ax, run_data.blocks, run_data.spawn, run_data.goal)
    render_heatmap(ax, fig, density_grid, grid_meta,
                   run_data.blocks, colormap=args.colormap)
    render_special_markers(ax, run_data.spawn, run_data.goal)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------

def _run_replay(run_data, run_name, args):
    theme_name = args.theme or "dark"
    t          = _THEMES.get(theme_name, _THEMES["dark"])

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(t["bg"])
    fig.canvas.manager.set_window_title(f"Replay — {run_name}")

    # Layout
    ax3d    = fig.add_subplot(111, projection="3d")
    ax3d.set_position([0.0, 0.15, 0.78, 0.83])

    info_ax = fig.add_axes([0.80, 0.15, 0.18, 0.83])
    ax3d.set_facecolor(t["bg"])
    info_ax.set_facecolor(t["panel_bg"])
    info_ax.axis("off")

    ax3d.view_init(elev=25, azim=-60)

    # Static world geometry
    render_world(ax3d, run_data.blocks)
    configure_axes(ax3d, run_data.blocks, run_data.spawn, run_data.goal)
    render_special_markers(ax3d, run_data.spawn, run_data.goal)

    _apply_theme(fig, ax3d, info_ax, theme_name)

    episodes     = run_data.episodes
    sorted_eps   = sorted(episodes.keys()) if episodes else []
    total_eps    = len(sorted_eps)
    ep_to_idx    = {ep: i for i, ep in enumerate(sorted_eps)}

    has_data = total_eps > 0

    # --- No-data graceful degradation ---
    if not has_data or run_data.source == "xml_only":
        msg = "No trajectory data available.\nWorld geometry rendered from XML."
        ax3d.text2D(0.5, 0.5, msg,
                    transform=ax3d.transAxes,
                    ha="center", va="center",
                    fontsize=12, color="#FFCC44")
        ax3d.set_title(f"World only — {run_name}", color=t["text"], pad=10)
        plt.tight_layout()
        plt.show()
        return

    # --- Sliders ---
    start_ep = args.episode if (args.episode and args.episode in episodes) else sorted_eps[0]

    ax_ep   = fig.add_axes([0.08, 0.08, 0.70, 0.03])
    ax_step = fig.add_axes([0.08, 0.03, 0.70, 0.03])

    init_steps = len(episodes[start_ep]["steps"])
    max_steps  = max(init_steps - 1, 0)

    ep_slider   = Slider(ax_ep,   "Episode", sorted_eps[0], sorted_eps[-1],
                         valinit=start_ep, valstep=1,
                         color=t["slider_ep"])
    step_slider = Slider(ax_step, "Step",    0,             max_steps,
                         valinit=0,         valstep=1,
                         color=t["slider_step"])

    for sl_ax in (ax_ep, ax_step):
        sl_ax.set_facecolor(t["panel_bg"])
        for spine in sl_ax.spines.values():
            spine.set_edgecolor(t["text"])
        sl_ax.tick_params(colors=t["text"])
        sl_ax.yaxis.label.set_color(t["text"])
        sl_ax.xaxis.label.set_color(t["text"])

    # --- Shared state ---
    state = {
        "current_ep":      start_ep,
        "current_steps":   episodes[start_ep]["steps"],
        "current_outcome": episodes[start_ep].get("outcome", "alive"),
        "trajectory_line": None,
        "dot":             None,
        "step_idx":        0,
    }

    def _draw_episode(ep):
        clear_trajectory(ax3d, state)
        ep_data = episodes.get(ep, {})
        steps   = ep_data.get("steps", [])
        outcome = ep_data.get("outcome", "alive")

        state["current_ep"]      = ep
        state["current_steps"]   = steps
        state["current_outcome"] = outcome
        state["step_idx"]        = 0

        if steps:
            line, dot = render_trajectory(ax3d, steps, outcome)
            state["trajectory_line"] = line
            state["dot"]             = dot
        else:
            state["trajectory_line"] = None
            state["dot"]             = None

        # Reset step slider range
        n = max(len(steps) - 1, 0)
        step_slider.valmax = n
        step_slider.ax.set_xlim(0, max(n, 1))
        step_slider.set_val(0)

        _update_info_panel(info_ax, state, run_data, run_name, theme_name, total_eps)
        fig.canvas.draw_idle()

    def on_episode_change(val):
        # Snap val to the nearest actual episode key
        requested = int(round(val))
        # Find closest available episode
        ep = min(sorted_eps, key=lambda e: abs(e - requested))
        if ep == state["current_ep"]:
            return
        _draw_episode(ep)

    def on_step_change(val):
        sidx = int(round(val))
        state["step_idx"] = sidx
        update_dot(ax3d, state, sidx)
        _update_info_panel(info_ax, state, run_data, run_name, theme_name, total_eps)
        fig.canvas.draw_idle()

    ep_slider.on_changed(on_episode_change)
    step_slider.on_changed(on_step_change)

    # --- Keyboard shortcuts ---
    def on_key(event):
        key = event.key
        steps = state["current_steps"]
        n     = max(len(steps) - 1, 0)
        sidx  = state["step_idx"]
        ei    = ep_to_idx.get(state["current_ep"], 0)

        if key == "right":
            step_slider.set_val(min(sidx + 1, n))
        elif key == "left":
            step_slider.set_val(max(sidx - 1, 0))
        elif key == "up":
            if ei + 1 < total_eps:
                ep_slider.set_val(sorted_eps[ei + 1])
        elif key == "down":
            if ei - 1 >= 0:
                ep_slider.set_val(sorted_eps[ei - 1])
        elif key == "home":
            step_slider.set_val(0)
        elif key == "end":
            step_slider.set_val(n)
        elif key == "r":
            ax3d.view_init(elev=25, azim=-60)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Draw the initial episode
    _draw_episode(start_ep)

    ax3d.set_title(f"Replay — {run_name}", color=t["text"], fontsize=10, pad=10)
    plt.show()


# ---------------------------------------------------------------------------
# Run selection helper
# ---------------------------------------------------------------------------

def _pick_run(run_arg):
    """
    Given --run argument, resolve to a single run prefix.
    If directory with multiple runs, prompt user to pick.
    """
    prefixes = discover_runs(run_arg)
    if not prefixes:
        print(f"[replay] No runs found at: {run_arg}")
        sys.exit(1)
    if len(prefixes) == 1:
        return prefixes[0]

    print("\nAvailable runs:")
    for i, p in enumerate(prefixes):
        print(f"  [{i+1}] {os.path.basename(p)}")
    while True:
        try:
            choice = input(f"\nSelect run [1–{len(prefixes)}]: ").strip()
            idx    = int(choice) - 1
            if 0 <= idx < len(prefixes):
                return prefixes[idx]
        except (ValueError, EOFError):
            pass
        print("  Invalid selection, try again.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_episode_list(s):
    """Parse '1,2,5-10' into a set of ints."""
    result = set()
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(token))
    return result


def build_parser():
    p = argparse.ArgumentParser(
        description="Replay and visualise agent trajectories from Minecraft RL training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run", required=True,
                   help="Path to a run prefix, trajectory CSV, or directory of runs.")
    p.add_argument("--heatmap", action="store_true",
                   help="Show aggregate visit-density heatmap instead of per-episode replay.")
    p.add_argument("--episode", type=int, default=None,
                   help="Episode number to start the replay at.")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated list or ranges, e.g. '1,5,10-20' (for heatmap filter).")
    p.add_argument("--outcome", type=str, default=None,
                   choices=list(OUTCOME_COLORS.keys()),
                   help="Filter episodes by outcome (for heatmap mode).")
    p.add_argument("--colormap", type=str, default="hot",
                   help="Matplotlib colormap name for heatmap (default: hot).")
    p.add_argument("--theme", type=str, default="dark",
                   choices=["dark", "light"],
                   help="Colour theme (default: dark).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Parse episode list if provided
    if args.episodes:
        args.episodes = _parse_episode_list(args.episodes)
    else:
        args.episodes = None

    # Resolve run
    run_prefix = _pick_run(args.run)
    run_name   = os.path.basename(run_prefix)

    print(f"[replay] Loading run: {run_prefix}")
    run_data = load_run(run_prefix)

    print(f"[replay] env={run_data.env_name}  "
          f"source={run_data.source}  "
          f"episodes={len(run_data.episodes)}  "
          f"blocks={len(run_data.blocks)}")

    if args.heatmap:
        _run_heatmap(run_data, run_name, args)
    else:
        _run_replay(run_data, run_name, args)


if __name__ == "__main__":
    main()
