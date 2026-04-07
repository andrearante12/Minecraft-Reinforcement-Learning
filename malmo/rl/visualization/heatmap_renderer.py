"""
visualization/heatmap_renderer.py
----------------------------------
Density accumulation and heatmap rendering for aggregate trajectory analysis.

The heatmap is projected onto the platform plane (minimum Y across all
episode positions) so it reads as a top-down density overlay.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Density accumulation
# ---------------------------------------------------------------------------

def accumulate_density(episodes, outcome_filter=None, resolution=0.25):
    """
    Bin all agent positions into a 3-D density grid.

    Parameters
    ----------
    episodes       : dict[int, {"steps": list[dict], "outcome": str}]
    outcome_filter : str or None  — if set, only include episodes with this outcome
    resolution     : float  — voxel size in world units

    Returns
    -------
    (density_grid, grid_meta)
    density_grid : np.ndarray shape (nx, ny, nz)
    grid_meta    : dict with keys x_min, y_min, z_min, resolution, shape
    """
    all_positions = []

    for ep_data in episodes.values():
        if outcome_filter and ep_data.get("outcome") != outcome_filter:
            continue
        for s in ep_data.get("steps", []):
            all_positions.append((s["x"], s["y"], s["z"]))

    if not all_positions:
        empty = np.zeros((1, 1, 1), dtype=np.float32)
        meta  = {"x_min": 0, "y_min": 0, "z_min": 0,
                 "resolution": resolution, "shape": (1, 1, 1)}
        return empty, meta

    positions = np.array(all_positions, dtype=np.float64)
    x_min = positions[:, 0].min()
    y_min = positions[:, 1].min()
    z_min = positions[:, 2].min()
    x_max = positions[:, 0].max()
    y_max = positions[:, 1].max()
    z_max = positions[:, 2].max()

    nx = max(int(np.ceil((x_max - x_min) / resolution)) + 1, 1)
    ny = max(int(np.ceil((y_max - y_min) / resolution)) + 1, 1)
    nz = max(int(np.ceil((z_max - z_min) / resolution)) + 1, 1)

    grid = np.zeros((nx, ny, nz), dtype=np.float32)

    for (px, py, pz) in all_positions:
        xi = int((px - x_min) / resolution)
        yi = int((py - y_min) / resolution)
        zi = int((pz - z_min) / resolution)
        xi = min(xi, nx - 1)
        yi = min(yi, ny - 1)
        zi = min(zi, nz - 1)
        grid[xi, yi, zi] += 1.0

    meta = {
        "x_min":      x_min,
        "y_min":      y_min,
        "z_min":      z_min,
        "resolution": resolution,
        "shape":      (nx, ny, nz),
    }
    return grid, meta


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def render_heatmap(ax, fig, density_grid, grid_meta, blocks, colormap="hot",
                   render_world_geom=True, show_colorbar=True):
    """
    Render a heatmap scatter overlay on the platform plane.

    Voxels with count > 0 are projected to a flat Y plane and scatter-plotted
    with size/alpha proportional to normalised density.  Blocks are rendered
    in wireframe-only mode so the geometry is still visible beneath the heatmap.

    Parameters
    ----------
    ax               : Axes3D
    fig              : Figure  (for colorbar)
    density_grid     : np.ndarray
    grid_meta        : dict
    blocks           : list[Block]
    colormap         : str
    render_world_geom: bool — if False, skip wireframe world rendering (use when
                       world is already drawn on the axes, e.g. the replay overlay)
    show_colorbar    : bool — if False, skip colorbar (avoids layout disruption
                       when used as an overlay on the replay view)

    Returns
    -------
    (scatter, colorbar) — either may be None
    """
    from visualization.world_renderer import render_world, configure_axes

    if render_world_geom:
        render_world(ax, blocks, wireframe_only=True)

    nx, ny, nz = density_grid.shape
    res  = grid_meta["resolution"]
    xm   = grid_meta["x_min"]
    ym   = grid_meta["y_min"]
    zm   = grid_meta["z_min"]

    max_count = density_grid.max()
    if max_count == 0:
        return None, None

    xs_plot, ys_plot, zs_plot, cs_plot, ss_plot = [], [], [], [], []

    # Project to the Y level with the highest total visit count (the standing
    # surface), rather than the minimum Y which is pulled down by fall positions.
    y_counts = density_grid.sum(axis=(0, 2))  # sum over x and z → shape (ny,)
    yi_peak  = int(np.argmax(y_counts))
    y_flat   = ym + yi_peak * res + res / 2

    for xi in range(nx):
        for zi in range(nz):
            col_total = density_grid[xi, :, zi].sum()
            if col_total == 0:
                continue
            world_x = xm + xi * res + res / 2
            world_z = zm + zi * res + res / 2
            norm    = float(col_total) / max_count
            xs_plot.append(world_x)
            ys_plot.append(y_flat)
            zs_plot.append(world_z)
            cs_plot.append(norm)
            ss_plot.append(max(5.0, 30.0 * norm))

    if not xs_plot:
        return None

    sc = ax.scatter(
        xs_plot, ys_plot, zs_plot,
        c=cs_plot,
        s=ss_plot,
        cmap=colormap,
        alpha=0.7,
        vmin=0, vmax=1,
    )

    cb = None
    if show_colorbar:
        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label="Relative visit density")
    return sc, cb
