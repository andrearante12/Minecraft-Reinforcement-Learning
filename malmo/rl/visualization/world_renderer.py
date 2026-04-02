"""
visualization/world_renderer.py
--------------------------------
Renders static Minecraft world geometry (block cubes) onto a 3D matplotlib axis.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Face generation
# ---------------------------------------------------------------------------

def _cube_faces(bx, by, bz):
    """Return the 6 quad faces for a unit cube at integer position (bx, by, bz)."""
    x0, x1 = bx,     bx + 1
    y0, y1 = by,     by + 1
    z0, z1 = bz,     bz + 1

    return [
        # bottom (y=y0)
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        # top (y=y1)
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        # front (z=z0)
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        # back (z=z1)
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        # left (x=x0)
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        # right (x=x1)
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]


def build_block_faces(blocks):
    """
    Flatten all cube faces from a list of Block namedtuples into a single list.

    Returns
    -------
    list of face quads (each face = list of 4 (x,y,z) tuples)
    """
    all_faces = []
    for b in blocks:
        all_faces.extend(_cube_faces(b.x, b.y, b.z))
    return all_faces


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_world(ax, blocks, wireframe_only=False):
    """
    Draw all blocks as a single Poly3DCollection.

    Parameters
    ----------
    ax            : Axes3D
    blocks        : list[Block]
    wireframe_only: if True, transparent faces (for heatmap mode)

    Returns
    -------
    Poly3DCollection or None (if no blocks)
    """
    if not blocks:
        return None

    faces = build_block_faces(blocks)

    if wireframe_only:
        collection = Poly3DCollection(
            faces,
            facecolor=(0, 0, 0, 0),
            edgecolor="#444444",
            linewidth=0.5,
            zsort="average",
        )
    else:
        collection = Poly3DCollection(
            faces,
            facecolor="#6B6B6B",
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.75,
            zsort="average",
        )

    ax.add_collection3d(collection)
    return collection


def configure_axes(ax, blocks, spawn=None, goal=None):
    """
    Set axis limits, aspect ratio, labels, and grid style.

    Computes a bounding box from blocks + spawn + goal with 1-block padding,
    then applies equal-scale box aspect.
    """
    xs, ys, zs = [], [], []

    for b in blocks:
        xs.extend([b.x, b.x + 1])
        ys.extend([b.y, b.y + 1])
        zs.extend([b.z, b.z + 1])

    for pt in (spawn, goal):
        if pt is not None:
            xs.append(pt[0])
            ys.append(pt[1])
            zs.append(pt[2])

    if not xs:
        return  # nothing to bound

    pad = 1.5
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    zmin, zmax = min(zs) - pad, max(zs) + pad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    dx = max(xmax - xmin, 0.1)
    dy = max(ymax - ymin, 0.1)
    dz = max(zmax - zmin, 0.1)
    ax.set_box_aspect([dx, dy, dz])

    # Hide pane fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Subtle grid
    ax.xaxis._axinfo["grid"].update({"linestyle": "--", "alpha": 0.3})
    ax.yaxis._axinfo["grid"].update({"linestyle": "--", "alpha": 0.3})
    ax.zaxis._axinfo["grid"].update({"linestyle": "--", "alpha": 0.3})

    ax.set_xlabel("X",          labelpad=6)
    ax.set_ylabel("Y (height)", labelpad=6)
    ax.set_zlabel("Z (forward)", labelpad=6)


def render_special_markers(ax, spawn=None, goal=None):
    """
    Draw spawn (green star) and goal (gold diamond) markers.

    Returns
    -------
    tuple (spawn_scatter, goal_scatter) — either may be None
    """
    sp_sc = None
    go_sc = None

    if spawn is not None:
        sp_sc = ax.scatter(
            [spawn[0]], [spawn[1]], [spawn[2]],
            marker="*", s=200, c="#00CC44",
            zorder=6, label="Spawn",
        )

    if goal is not None:
        go_sc = ax.scatter(
            [goal[0]], [goal[1]], [goal[2]],
            marker="D", s=150, c="#FFD700",
            zorder=6, label="Goal",
        )

    return sp_sc, go_sc
