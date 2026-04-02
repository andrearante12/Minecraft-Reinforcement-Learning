"""
visualization/data_loader.py
-----------------------------
CSV/XML parsing and run discovery for the replay tool.

Supports two data sources:
  1. *_trajectories.csv  — full per-step data (future logger output)
  2. *_episodes.csv only — falls back to XML geometry with no trajectory data
"""

import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Block = namedtuple("Block", ["x", "y", "z", "block_type"])

RunData = namedtuple("RunData", [
    "env_name",   # str
    "blocks",     # list[Block]
    "spawn",      # (x, y, z)
    "goal",       # (x, y, z) or None
    "episodes",   # dict[int, {"steps": list[dict], "outcome": str}]
    "source",     # "csv" | "xml_only"
])

# ---------------------------------------------------------------------------
# Lightweight config registry — no ML/PyTorch imports
# ---------------------------------------------------------------------------

def _build_config_registry():
    """Import config classes without pulling in train.py (and therefore torch)."""
    registry = {}
    # Add the rl directory to sys.path so config imports work standalone
    rl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)

    try:
        from training.configs.simple_jump_cfg   import SimpleJumpCFG
        registry["simple_jump"] = SimpleJumpCFG
    except ImportError:
        pass
    try:
        from training.configs.one_block_gap_cfg  import OneBlockGapCFG
        registry["one_block_gap"] = OneBlockGapCFG
    except ImportError:
        pass
    try:
        from training.configs.three_block_gap_cfg import ThreeBlockGapCFG
        registry["three_block_gap"] = ThreeBlockGapCFG
    except ImportError:
        pass
    try:
        from training.configs.multi_jump_course_cfg import MultiJumpCourseCFG
        registry["multi_jump_course"] = MultiJumpCourseCFG
    except ImportError:
        pass
    try:
        from training.configs.diagonal_small_cfg import DiagonalSmallCFG
        registry["diagonal_small"] = DiagonalSmallCFG
    except ImportError:
        pass
    try:
        from training.configs.diagonal_medium_cfg import DiagonalMediumCFG
        registry["diagonal_medium"] = DiagonalMediumCFG
    except ImportError:
        pass
    try:
        from training.configs.vertical_small_cfg import VerticalSmallCFG
        registry["vertical_small"] = VerticalSmallCFG
    except ImportError:
        pass

    return registry


CONFIG_REGISTRY = _build_config_registry()

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

_TRAJ_RE = re.compile(r'^(.+_\d{8}_\d{6})_trajectories\.csv$')
_EP_RE   = re.compile(r'^(.+_\d{8}_\d{6})_episodes\.csv$')


def discover_runs(path):
    """
    Scan *path* (file or directory) for run prefixes.

    Returns a sorted list of absolute run prefix strings.
    If path is a file, returns [path-without-suffix] or [path] unchanged.
    If path is a directory, scans for *_trajectories.csv first; if none
    found, falls back to *_episodes.csv.
    """
    path = os.path.abspath(path)

    if os.path.isfile(path):
        # Strip known suffixes, return as single-item list
        for suffix in ("_trajectories.csv", "_episodes.csv", ".csv"):
            if path.endswith(suffix):
                return [path[: -len(suffix)]]
        return [path]

    if not os.path.isdir(path):
        return [path]  # treat as prefix

    entries = os.listdir(path)
    prefixes = set()

    for name in entries:
        m = _TRAJ_RE.match(name)
        if m:
            prefixes.add(os.path.join(path, m.group(1)))

    if not prefixes:
        for name in entries:
            m = _EP_RE.match(name)
            if m:
                prefixes.add(os.path.join(path, m.group(1)))

    return sorted(prefixes)


# ---------------------------------------------------------------------------
# Trajectory CSV loader
# ---------------------------------------------------------------------------

def load_trajectory_csv(path):
    """
    Parse a *_trajectories.csv file.

    Header lines (starting with '#') may contain:
        # env: <name>
        # block: x,y,z,type
        # spawn: x,y,z
        # goal: x,y,z

    Returns:
        (blocks, env_name, spawn, goal, episodes_dict)
        episodes_dict: {ep_int: {"steps": [step_dict, ...], "outcome": str}}
    """
    blocks   = []
    env_name = "unknown"
    spawn    = None
    goal     = None
    episodes = {}

    with open(path, newline="") as fh:
        all_lines   = fh.readlines()

    fieldnames  = None
    data_lines  = []

    for raw_line in all_lines:
        line = raw_line.rstrip("\n")

        if fieldnames is None:
            if line.startswith("#"):
                content = line[1:].strip()
                if content.startswith("env:"):
                    env_name = content[4:].strip()
                elif content.startswith("block:"):
                    parts = content[6:].strip().split(",")
                    if len(parts) == 4:
                        bx, by, bz, bt = parts
                        blocks.append(Block(int(bx), int(by), int(bz), bt.strip()))
                elif content.startswith("spawn:"):
                    coords = content[6:].strip().split(",")
                    if len(coords) == 3:
                        spawn = tuple(float(c) for c in coords)
                elif content.startswith("goal:"):
                    coords = content[5:].strip().split(",")
                    if len(coords) == 3:
                        goal = tuple(float(c) for c in coords)
            else:
                # First non-# line is the CSV header
                fieldnames = [f.strip() for f in line.split(",")]
        else:
            data_lines.append(line)

    if fieldnames is not None:
        reader = csv.DictReader(data_lines, fieldnames=fieldnames)
        for row in reader:
            try:
                ep  = int(row["episode"])
                out = row.get("outcome", "alive").strip()
            except (KeyError, ValueError):
                continue

            step = {
                "step":      int(row.get("step", 0)),
                "x":         float(row.get("x", 0)),
                "y":         float(row.get("y", 0)),
                "z":         float(row.get("z", 0)),
                "yaw":       float(row.get("yaw", 0)),
                "pitch":     float(row.get("pitch", 0)),
                "on_ground": row.get("on_ground", "1") in ("1", "True", "true"),
                "action":    row.get("action", ""),
                "reward":    float(row.get("reward", 0)),
                "done":      row.get("done", "False") in ("1", "True", "true"),
                "outcome":   out,
                "env":       row.get("env", env_name).strip(),
            }

            if ep not in episodes:
                episodes[ep] = {"steps": [], "outcome": "alive"}
            episodes[ep]["steps"].append(step)

            # Last step of the episode determines the outcome
            if step["done"] or out not in ("alive",):
                episodes[ep]["outcome"] = out

    return blocks, env_name, spawn, goal, episodes


# ---------------------------------------------------------------------------
# XML geometry loader
# ---------------------------------------------------------------------------

_MALMO_NS = {"m": "http://ProjectMalmo.microsoft.com"}


def load_xml_geometry(xml_path):
    """
    Parse a Malmo mission XML and extract block geometry and spawn position.

    Returns:
        (blocks, spawn)
        blocks: list[Block]
        spawn:  (x, y, z) or None
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    blocks = []
    spawn  = None

    # Try namespaced search first, fall back to no-namespace
    def find_all(tag):
        results = root.findall(f".//m:{tag}", _MALMO_NS)
        if not results:
            results = root.findall(f".//{tag}")
        return results

    def find_first(tag):
        results = find_all(tag)
        return results[0] if results else None

    # --- Blocks from DrawCuboid ---
    for cuboid in find_all("DrawCuboid"):
        try:
            x1 = int(cuboid.get("x1"))
            y1 = int(cuboid.get("y1"))
            z1 = int(cuboid.get("z1"))
            x2 = int(cuboid.get("x2"))
            y2 = int(cuboid.get("y2"))
            z2 = int(cuboid.get("z2"))
            btype = cuboid.get("type", "stone")
        except (TypeError, ValueError):
            continue

        for bx in range(min(x1, x2), max(x1, x2) + 1):
            for by in range(min(y1, y2), max(y1, y2) + 1):
                for bz in range(min(z1, z2), max(z1, z2) + 1):
                    blocks.append(Block(bx, by, bz, btype))

    # --- Spawn from Placement ---
    placement = find_first("Placement")
    if placement is not None:
        try:
            px = float(placement.get("x", 0))
            py = float(placement.get("y", 0))
            pz = float(placement.get("z", 0))
            spawn = (px, py, pz)
        except (TypeError, ValueError):
            pass

    return blocks, spawn


# ---------------------------------------------------------------------------
# Env name → XML path resolution
# ---------------------------------------------------------------------------

def _resolve_xml_from_env(env_name):
    """Look up MISSION_FILE from the config registry."""
    cfg = CONFIG_REGISTRY.get(env_name)
    if cfg is None:
        return None
    mf = getattr(cfg, "MISSION_FILE", None)
    if mf and os.path.isfile(mf):
        return mf
    return None


def _infer_env_name(run_prefix):
    """
    Infer environment name from the run prefix stem.

    e.g. '/logs/simple_jump_ppo_20260401_143022' → 'simple_jump'
    Strategy: try longest matching key in CONFIG_REGISTRY.
    """
    stem = os.path.basename(run_prefix)
    best = None
    for key in CONFIG_REGISTRY:
        if stem.startswith(key):
            if best is None or len(key) > len(best):
                best = key
    return best or "unknown"


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_run(run_prefix):
    """
    Load a run, trying *_trajectories.csv first, then XML-only fallback.

    Parameters
    ----------
    run_prefix : str
        Path prefix, e.g. 'logs/simple_jump_ppo_20260401_143022'.
        May or may not include the file extension.

    Returns
    -------
    RunData
    """
    traj_path = run_prefix + "_trajectories.csv"
    if os.path.isfile(traj_path):
        blocks, env_name, spawn, goal, episodes = load_trajectory_csv(traj_path)

        # If XML header didn't supply blocks, try XML
        if not blocks:
            xml_path = _resolve_xml_from_env(env_name)
            if xml_path:
                xml_blocks, xml_spawn = load_xml_geometry(xml_path)
                blocks = xml_blocks
                if spawn is None:
                    spawn = xml_spawn

        # Supplement spawn/goal from config if missing
        cfg = CONFIG_REGISTRY.get(env_name)
        if cfg:
            if spawn is None:
                spawn = getattr(cfg, "SPAWN", None)
            if goal is None:
                goal = getattr(cfg, "GOAL_POS", None)

        return RunData(
            env_name=env_name,
            blocks=blocks,
            spawn=spawn,
            goal=goal,
            episodes=episodes,
            source="csv",
        )

    # --- Fallback: XML-only ---
    env_name = _infer_env_name(run_prefix)
    xml_path = _resolve_xml_from_env(env_name)

    blocks = []
    spawn  = None
    goal   = None

    if xml_path:
        blocks, spawn = load_xml_geometry(xml_path)

    cfg = CONFIG_REGISTRY.get(env_name)
    if cfg:
        if spawn is None:
            spawn = getattr(cfg, "SPAWN", None)
        goal = getattr(cfg, "GOAL_POS", None)

    return RunData(
        env_name=env_name,
        blocks=blocks,
        spawn=spawn,
        goal=goal,
        episodes={},
        source="xml_only",
    )
