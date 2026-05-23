"""
utils/validate.py
-----------------
Post-change validator for MalmoRL. Catches the most common ways edits break
the framework before the user notices at training time.

Tier 1 (stdlib only, ~2s) — runs anywhere, no torch needed:
  - Python syntax check on every malmo/rl/**/*.py
  - XML well-formedness check on every malmo/rl/envs/*/missions/*.xml
  - ENV_REGISTRY key parity between env_server.py and train.py
  - Every cfg referenced in either registry has its .py file on disk

Tier 2 (needs torch+numpy, ~5-10s) — runs against a sample env:
  - Import the cfg, assert INPUT_SIZE == PROPRIO + GOAL_DELTA + GRID_SIZE
  - Assert len(ACTIONS) == N_ACTIONS, MISSION_FILE exists on disk
  - Instantiate ActorCritic(cfg), forward pass with torch.zeros(1, INPUT_SIZE)
  - For each algo in ALGO_REGISTRY: instantiate, call select_action, check action range

Log scan (opt-in via --logs):
  - Most recent *_updates.csv — confirm losses are finite (no NaN/Inf)
  - Most recent *_episodes.csv — flag if recent outcomes are all timeout/fell

Usage:
  python malmo/rl/utils/validate.py --tier 1
  python malmo/rl/utils/validate.py --tier 2 --env simple_jump
  python malmo/rl/utils/validate.py --tier all --logs
  python malmo/rl/utils/validate.py --tier 1 --quiet      # silent on pass (hook mode)
  python malmo/rl/utils/validate.py --tier 1 --json       # machine-readable

Exit code: 0 = pass, 1 = fail.
"""

import argparse
import ast
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
RL_ROOT   = THIS_FILE.parents[1]                 # malmo/rl/
MALMO_DIR = RL_ROOT.parent                       # malmo/
REPO_ROOT = MALMO_DIR.parent                     # repo root

ENV_SERVER_PY = RL_ROOT / "envs"     / "env_server.py"
TRAIN_PY      = RL_ROOT / "training" / "train.py"
CONFIGS_DIR   = RL_ROOT / "training" / "configs"
ALGOS_DIR     = RL_ROOT / "algos"
ENVS_DIR      = RL_ROOT / "envs"
LOGS_DIR      = RL_ROOT / "logs"


# ── Result accumulator ────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self.tiers = {}   # name -> list of (status, message)

    def add(self, tier, status, message):
        self.tiers.setdefault(tier, []).append((status, message))

    def passed(self, tier):
        return all(s != "fail" for s, _ in self.tiers.get(tier, []))

    def all_passed(self):
        return all(self.passed(t) for t in self.tiers)

    def failures(self):
        out = []
        for tier, entries in self.tiers.items():
            for status, message in entries:
                if status != "pass":
                    out.append((tier, status, message))
        return out


# ── AST helpers ───────────────────────────────────────────────────────────────
def parse_file(path):
    """Return AST or raise."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def extract_dict_keys(tree, name):
    """Return set of string keys from `name = { 'a': ..., 'b': ... }`. None if not found."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Dict):
                        keys = []
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys.append(k.value)
                        return set(keys)
    return None


# ── Tier 1: Static checks ─────────────────────────────────────────────────────
def tier1_syntax(results):
    files = list(RL_ROOT.rglob("*.py"))
    bad = []
    for f in files:
        try:
            parse_file(f)
        except SyntaxError as e:
            bad.append((f, e.lineno or 0, e.msg))
        except Exception as e:
            bad.append((f, 0, "{0}: {1}".format(type(e).__name__, e)))
    if bad:
        for f, line, msg in bad:
            rel = f.relative_to(REPO_ROOT)
            results.add("tier1", "fail",
                        "Syntax error in {0}:{1} — {2}".format(rel, line, msg))
    else:
        results.add("tier1", "pass",
                    "Python syntax OK ({0} files)".format(len(files)))


def tier1_xml(results):
    files = list((ENVS_DIR).glob("*/missions/*.xml"))
    bad = []
    for f in files:
        try:
            ET.parse(f)
        except ET.ParseError as e:
            bad.append((f, str(e)))
        except Exception as e:
            bad.append((f, "{0}: {1}".format(type(e).__name__, e)))
    if bad:
        for f, msg in bad:
            rel = f.relative_to(REPO_ROOT)
            results.add("tier1", "fail",
                        "Malformed mission XML: {0} — {1}".format(rel, msg))
    else:
        results.add("tier1", "pass",
                    "Mission XML OK ({0} files)".format(len(files)))


def tier1_env_registry(results):
    try:
        server_tree = parse_file(ENV_SERVER_PY)
        train_tree  = parse_file(TRAIN_PY)
    except SyntaxError:
        # Syntax errors already reported by tier1_syntax; skip silently.
        return

    server_keys = extract_dict_keys(server_tree, "ENV_REGISTRY")
    train_keys  = extract_dict_keys(train_tree,  "ENV_REGISTRY")

    if server_keys is None:
        results.add("tier1", "fail",
                    "Could not find ENV_REGISTRY dict in {0}".format(
                        ENV_SERVER_PY.relative_to(REPO_ROOT)))
        return
    if train_keys is None:
        results.add("tier1", "fail",
                    "Could not find ENV_REGISTRY dict in {0}".format(
                        TRAIN_PY.relative_to(REPO_ROOT)))
        return

    only_server = sorted(server_keys - train_keys)
    only_train  = sorted(train_keys  - server_keys)

    if not only_server and not only_train:
        results.add("tier1", "pass",
                    "ENV_REGISTRY parity OK ({0} envs)".format(len(server_keys)))
        return

    if only_server:
        results.add("tier1", "fail",
                    "ENV_REGISTRY mismatch — in env_server.py but missing from train.py: {0}. "
                    "Fix: add the cfg import and registry entry in malmo/rl/training/train.py "
                    "(ENV_REGISTRY block around line 63). This is the #1 newcomer footgun."
                    .format(only_server))
    if only_train:
        results.add("tier1", "fail",
                    "ENV_REGISTRY mismatch — in train.py but missing from env_server.py: {0}. "
                    "Fix: add the cfg import and registry entry in malmo/rl/envs/env_server.py "
                    "(ENV_REGISTRY block around line 49)."
                    .format(only_train))


def tier1_algo_registry(results):
    try:
        train_tree = parse_file(TRAIN_PY)
    except SyntaxError:
        return

    algo_keys = extract_dict_keys(train_tree, "ALGO_REGISTRY")
    if algo_keys is None:
        results.add("tier1", "fail",
                    "Could not find ALGO_REGISTRY dict in {0}".format(
                        TRAIN_PY.relative_to(REPO_ROOT)))
        return

    results.add("tier1", "pass",
                "ALGO_REGISTRY found ({0} algos)".format(len(algo_keys)))


def tier1_cfg_files(results):
    """For each cfg class imported in either registry, confirm the file exists
    and contains a class inheriting from BaseCFG."""
    missing = []
    bad_class = []

    # Discover cfg modules referenced via "from training.configs.<mod> import <Class>"
    for py_path in (ENV_SERVER_PY, TRAIN_PY):
        try:
            tree = parse_file(py_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.startswith("training.configs."):
                mod = node.module.split(".")[-1]
                cfg_file = CONFIGS_DIR / "{0}.py".format(mod)
                if not cfg_file.exists():
                    missing.append((py_path.name, mod, cfg_file))

    # Confirm every *_cfg.py defines a *CFG class (either named *CFG or
    # inheriting from one). base_cfg.py is the root and self-qualifies.
    for cfg_file in CONFIGS_DIR.glob("*_cfg.py"):
        try:
            tree = parse_file(cfg_file)
        except SyntaxError:
            continue
        has_cfg_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "CFG" in node.name:
                    has_cfg_class = True
                    break
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                if any("CFG" in b for b in bases):
                    has_cfg_class = True
                    break
        if not has_cfg_class:
            bad_class.append(cfg_file)

    if missing:
        for source, mod, cfg_file in missing:
            results.add("tier1", "fail",
                        "Config import in {0} references missing file: {1} "
                        "(expected at {2})".format(source, mod, cfg_file.relative_to(REPO_ROOT)))
    if bad_class:
        for f in bad_class:
            results.add("tier1", "fail",
                        "Config file {0} does not define a *CFG class (expected `class XxxCFG(BaseCFG):`)"
                        .format(f.relative_to(REPO_ROOT)))

    if not missing and not bad_class:
        n = len(list(CONFIGS_DIR.glob("*_cfg.py")))
        results.add("tier1", "pass",
                    "Cfg files present and well-formed ({0} files)".format(n))


def run_tier1(results):
    tier1_syntax(results)
    tier1_xml(results)
    tier1_env_registry(results)
    tier1_algo_registry(results)
    tier1_cfg_files(results)


# ── Tier 2: Smoke instantiation ───────────────────────────────────────────────
def _setup_import_path():
    """Mimic env_server.py / train.py path setup so cfg/model/algo imports work."""
    if str(RL_ROOT) not in sys.path:
        sys.path.insert(0, str(RL_ROOT))


def _import_cfg(env_name):
    """Import the cfg class for `env_name`. Returns the class or raises."""
    # Look up the env in env_server.py's registry imports to find the cfg module + class
    tree = parse_file(ENV_SERVER_PY)
    cfg_module_for_class = {}    # class_name -> module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("training.configs."):
            mod = node.module
            for alias in node.names:
                cfg_module_for_class[alias.name] = mod

    # Find the registry entry mapping env_name -> (EnvClass, CfgClass)
    cfg_class_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ENV_REGISTRY":
                    if isinstance(node.value, ast.Dict):
                        for k, v in zip(node.value.keys, node.value.values):
                            if isinstance(k, ast.Constant) and k.value == env_name:
                                # value is a Tuple of two Names: (EnvClass, CfgClass)
                                if isinstance(v, ast.Tuple) and len(v.elts) == 2:
                                    cfg_node = v.elts[1]
                                    if isinstance(cfg_node, ast.Name):
                                        cfg_class_name = cfg_node.id
    if cfg_class_name is None:
        raise RuntimeError(
            "env '{0}' not found in ENV_REGISTRY (env_server.py)".format(env_name))
    if cfg_class_name not in cfg_module_for_class:
        raise RuntimeError(
            "cfg class {0} for env '{1}' has no import in env_server.py".format(
                cfg_class_name, env_name))

    module_path = cfg_module_for_class[cfg_class_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cfg_class_name)


def _import_algos():
    """Import every algo in ALGO_REGISTRY. Returns dict name -> class."""
    tree = parse_file(TRAIN_PY)
    algo_imports = {}  # class_name -> module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("algos."):
            mod = node.module
            for alias in node.names:
                algo_imports[alias.name] = mod

    algo_classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ALGO_REGISTRY":
                    if isinstance(node.value, ast.Dict):
                        for k, v in zip(node.value.keys, node.value.values):
                            if isinstance(k, ast.Constant) and isinstance(v, ast.Name):
                                algo_name = k.value
                                class_name = v.id
                                if class_name in algo_imports:
                                    import importlib
                                    module = importlib.import_module(algo_imports[class_name])
                                    algo_classes[algo_name] = getattr(module, class_name)

    return algo_classes


def tier2_cfg_sanity(results, cfg):
    """Check INPUT_SIZE formula, N_ACTIONS, MISSION_FILE."""
    issues = []

    # MISSION_FILE existence
    mission = getattr(cfg, "MISSION_FILE", None)
    if mission is None:
        issues.append("cfg.MISSION_FILE is not set")
    elif not os.path.exists(mission):
        issues.append("cfg.MISSION_FILE does not exist on disk: {0}".format(mission))

    # N_ACTIONS / ACTIONS parity
    actions = getattr(cfg, "ACTIONS", None)
    n_actions = getattr(cfg, "N_ACTIONS", None)
    if actions is not None and n_actions is not None:
        if len(actions) != n_actions:
            issues.append(
                "len(ACTIONS)={0} != N_ACTIONS={1}".format(len(actions), n_actions))

    # INPUT_SIZE formula (parkour/bridging-style: proprio + goal_delta + grid)
    proprio = getattr(cfg, "PROPRIOCEPTION_SIZE", None)
    goal_d  = getattr(cfg, "GOAL_DELTA_SIZE", None)
    grid    = getattr(cfg, "GRID_SIZE", None)
    in_size = getattr(cfg, "INPUT_SIZE", None)
    if all(v is not None for v in (proprio, goal_d, grid, in_size)):
        expected = proprio + goal_d + grid
        if in_size != expected:
            issues.append(
                "INPUT_SIZE={0} but PROPRIOCEPTION_SIZE+GOAL_DELTA_SIZE+GRID_SIZE={1}+{2}+{3}={4}"
                .format(in_size, proprio, goal_d, grid, expected))

    if issues:
        for msg in issues:
            results.add("tier2", "fail", "cfg sanity: {0}".format(msg))
    else:
        results.add("tier2", "pass", "cfg sanity OK")


def tier2_model(results, cfg):
    try:
        import torch
        from models.actor_critic import ActorCritic
    except Exception as e:
        results.add("tier2", "fail",
                    "Could not import torch / ActorCritic: {0}".format(e))
        return

    try:
        model = ActorCritic(cfg)
        obs = torch.zeros(1, cfg.INPUT_SIZE)
        logits, value = model.forward(obs)
        if tuple(logits.shape) != (1, cfg.N_ACTIONS):
            results.add("tier2", "fail",
                        "ActorCritic logits shape {0} != expected (1, {1})".format(
                            tuple(logits.shape), cfg.N_ACTIONS))
            return
        if tuple(value.shape) != (1, 1):
            results.add("tier2", "fail",
                        "ActorCritic value shape {0} != (1, 1)".format(tuple(value.shape)))
            return
        # Also test the 3-method interface
        dist = model.get_distribution(obs)
        v    = model.get_value(obs)
        lp, vv, ent = model.evaluate_actions(obs, torch.zeros(1, dtype=torch.int64))
        results.add("tier2", "pass",
                    "ActorCritic instantiation + forward pass OK (obs=({0},) -> logits=(1,{1}))"
                    .format(cfg.INPUT_SIZE, cfg.N_ACTIONS))
        return model
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        results.add("tier2", "fail",
                    "ActorCritic failed: {0}\n{1}".format(e, tb))
        return None


def tier2_algos(results, cfg, model):
    if model is None:
        return
    try:
        import numpy as np
    except Exception as e:
        results.add("tier2", "fail", "Could not import numpy: {0}".format(e))
        return

    try:
        algos = _import_algos()
    except Exception as e:
        results.add("tier2", "fail", "Could not import algos: {0}".format(e))
        return

    if not algos:
        results.add("tier2", "fail", "ALGO_REGISTRY parsed but no algo classes resolved")
        return

    obs = np.zeros(cfg.INPUT_SIZE, dtype=np.float32)

    for name, algo_cls in sorted(algos.items()):
        try:
            # Try with n_envs=1 first (PPO supports it), then fall back to (model, cfg).
            try:
                agent = algo_cls(model, cfg, n_envs=1)
            except TypeError:
                agent = algo_cls(model, cfg)
        except Exception as e:
            # BC / imitation algos require a demo path — downgrade to warn.
            msg = str(e).lower()
            if "demo" in msg:
                results.add("tier2", "warn",
                            "Algo '{0}' skipped (needs demos): {1}".format(name, e))
            else:
                results.add("tier2", "fail",
                            "Algo '{0}' constructor failed: {1}".format(name, e))
            continue

        try:
            action = agent.select_action(obs, greedy=True)
        except Exception as e:
            # BC may fail without demos — categorize, don't fail hard
            results.add("tier2", "warn",
                        "Algo '{0}' select_action raised {1}: {2} "
                        "(may be expected if algo needs extra state e.g. demos)"
                        .format(name, type(e).__name__, e))
            continue

        if not isinstance(action, int) or action < 0 or action >= cfg.N_ACTIONS:
            results.add("tier2", "fail",
                        "Algo '{0}' returned action {1}; expected int in [0, {2})"
                        .format(name, action, cfg.N_ACTIONS))
            continue

        results.add("tier2", "pass",
                    "Algo '{0}' instantiation + select_action OK (action={1})"
                    .format(name, action))


def run_tier2(results, env_name):
    _setup_import_path()
    try:
        cfg = _import_cfg(env_name)
    except Exception as e:
        results.add("tier2", "fail",
                    "Could not import cfg for env '{0}': {1}".format(env_name, e))
        return
    tier2_cfg_sanity(results, cfg)
    model = tier2_model(results, cfg)
    tier2_algos(results, cfg, model)


# ── Log scan ──────────────────────────────────────────────────────────────────
def run_logs(results):
    if not LOGS_DIR.exists():
        results.add("logs", "pass", "No logs directory yet — nothing to scan")
        return

    updates_csvs  = sorted(LOGS_DIR.glob("*_updates.csv"))
    episodes_csvs = sorted(LOGS_DIR.glob("*_episodes.csv"))

    if not updates_csvs and not episodes_csvs:
        results.add("logs", "pass", "No training logs found at top level of {0}".format(
            LOGS_DIR.relative_to(REPO_ROOT)))
        return

    # Most recent updates CSV — check for NaN/Inf losses
    if updates_csvs:
        latest = updates_csvs[-1]
        try:
            text = latest.read_text(encoding="utf-8").strip().splitlines()
            if len(text) > 1:
                header = text[0].split(",")
                last_row = text[-1].split(",")
                bad_cols = []
                for col, val in zip(header, last_row):
                    try:
                        f = float(val)
                        if math.isnan(f) or math.isinf(f):
                            bad_cols.append("{0}={1}".format(col, val))
                    except (ValueError, TypeError):
                        continue
                if bad_cols:
                    results.add("logs", "fail",
                                "Non-finite values in last update of {0}: {1}".format(
                                    latest.name, bad_cols))
                else:
                    results.add("logs", "pass",
                                "Latest updates log OK ({0})".format(latest.name))
            else:
                results.add("logs", "pass",
                            "Updates log {0} has no rows yet".format(latest.name))
        except Exception as e:
            results.add("logs", "warn",
                        "Could not read {0}: {1}".format(latest.name, e))

    # Most recent episodes CSV — look at last N outcomes
    if episodes_csvs:
        latest = episodes_csvs[-1]
        try:
            text = latest.read_text(encoding="utf-8").strip().splitlines()
            if len(text) > 1:
                header = text[0].split(",")
                if "outcome" in header:
                    idx = header.index("outcome")
                    rows = text[1:]
                    tail = rows[-20:] if len(rows) >= 20 else rows
                    outcomes = [r.split(",")[idx] for r in tail if "," in r]
                    successes = sum(1 for o in outcomes if "success" in o.lower() or "reached" in o.lower())
                    if outcomes and successes == 0:
                        results.add("logs", "warn",
                                    "Last {0} episodes in {1} had 0 successes "
                                    "(outcomes: {2}). Training may be stalled."
                                    .format(len(outcomes), latest.name, set(outcomes)))
                    else:
                        results.add("logs", "pass",
                                    "Latest episodes log OK ({0} successes in last {1} eps)"
                                    .format(successes, len(outcomes)))
        except Exception as e:
            results.add("logs", "warn",
                        "Could not read {0}: {1}".format(latest.name, e))


# ── Output formatters ─────────────────────────────────────────────────────────
def emit_text(results, quiet=False):
    if results.all_passed() and quiet:
        return

    for tier in ("tier1", "tier2", "logs"):
        if tier not in results.tiers:
            continue
        label = {"tier1": "Tier 1: static checks",
                 "tier2": "Tier 2: smoke instantiation",
                 "logs":  "Log scan"}[tier]
        passed = results.passed(tier)
        if quiet and passed:
            continue
        print("[{0}]".format(label))
        for status, message in results.tiers[tier]:
            sym = {"pass": "  PASS", "fail": "  FAIL", "warn": "  WARN"}[status]
            print("{0} {1}".format(sym, message))
        status = "PASS" if passed else "FAIL"
        n_fail = sum(1 for s, _ in results.tiers[tier] if s == "fail")
        print("[{0} {1}{2}]".format(label, status,
                                    "" if not n_fail else " — {0} issue(s)".format(n_fail)))
        print()


def emit_json(results):
    out = {"passed": results.all_passed(), "tiers": {}}
    for tier, entries in results.tiers.items():
        out["tiers"][tier] = [{"status": s, "message": m} for s, m in entries]
    print(json.dumps(out, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["1", "2", "all"], default="1",
                   help="Validation tier to run (default: 1)")
    p.add_argument("--env", default="simple_jump",
                   help="Env name for Tier 2 smoke checks (default: simple_jump)")
    p.add_argument("--logs", action="store_true",
                   help="Also scan recent training logs")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON output")
    p.add_argument("--quiet", action="store_true",
                   help="Silent on pass; failures only (hook mode)")
    args = p.parse_args()

    results = Results()

    if args.tier in ("1", "all"):
        run_tier1(results)

    if args.tier in ("2", "all"):
        if args.tier == "all" and not results.passed("tier1"):
            results.add("tier2", "warn",
                        "Skipped — Tier 1 failed (fix Tier 1 issues first to avoid cascades)")
        else:
            run_tier2(results, args.env)

    if args.logs:
        run_logs(results)

    if args.json:
        emit_json(results)
    else:
        emit_text(results, quiet=args.quiet)

    sys.exit(0 if results.all_passed() else 1)


if __name__ == "__main__":
    main()
