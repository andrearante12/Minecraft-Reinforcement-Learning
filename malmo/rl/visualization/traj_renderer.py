"""
visualization/traj_renderer.py
-------------------------------
Draws agent trajectories, animated position dots, and manages rendering state
for interactive episode replay.
"""

# ---------------------------------------------------------------------------
# Outcome → colour mapping
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {
    "landed":   "#00CC44",
    "fell":     "#EE3333",
    "timeout":  "#888888",
    "near_miss":"#FFCC00",
    "alive":    "#4488FF",
}

_DEFAULT_COLOR = "#4488FF"


def _outcome_color(outcome):
    return OUTCOME_COLORS.get(outcome, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Trajectory rendering
# ---------------------------------------------------------------------------

def render_trajectory(ax, steps, outcome):
    """
    Draw a full trajectory line for a list of step dicts.

    Parameters
    ----------
    ax      : Axes3D
    steps   : list[dict]  each dict has "x", "y", "z" keys
    outcome : str

    Returns
    -------
    (line, dot)  — Line3D and initial scatter PathCollection
    """
    if not steps:
        return None, None

    xs = [s["x"] for s in steps]
    ys = [s["y"] for s in steps]
    zs = [s["z"] for s in steps]

    color = _outcome_color(outcome)

    line, = ax.plot(xs, ys, zs,
                    color=color,
                    alpha=0.8,
                    linewidth=1.5)

    dot = ax.scatter(
        [xs[0]], [ys[0]], [zs[0]],
        s=80, c=color, zorder=5,
    )

    return line, dot


def update_dot(ax, state, step_idx):
    """
    Move the animated dot to *step_idx* within the current episode.

    Removes the old scatter artist and creates a new one to avoid
    mutating private 3D internals.

    Parameters
    ----------
    ax       : Axes3D
    state    : dict  with keys "dot", "current_steps", "current_outcome"
    step_idx : int
    """
    steps = state.get("current_steps", [])
    if not steps:
        return

    step_idx = max(0, min(step_idx, len(steps) - 1))
    s = steps[step_idx]

    old_dot = state.get("dot")
    if old_dot is not None:
        try:
            old_dot.remove()
        except Exception:
            pass

    outcome = state.get("current_outcome", "alive")
    color   = _outcome_color(outcome)

    state["dot"] = ax.scatter(
        [s["x"]], [s["y"]], [s["z"]],
        s=80, c=color, zorder=5,
    )


def clear_trajectory(ax, state):
    """
    Remove the current trajectory line and dot from the axes.

    Parameters
    ----------
    ax    : Axes3D
    state : dict  with keys "trajectory_line" and "dot"
    """
    line = state.get("trajectory_line")
    if line is not None:
        try:
            line.remove()
        except Exception:
            pass
        state["trajectory_line"] = None

    dot = state.get("dot")
    if dot is not None:
        try:
            dot.remove()
        except Exception:
            pass
        state["dot"] = None
