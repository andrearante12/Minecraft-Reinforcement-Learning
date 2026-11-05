"""
evaluation/compare_sample_efficiency.py
---------------------------------------
Milestone-#1 metrics + Experiment 1 (Dreamer vs PPO sample efficiency), straight
from the logger's *_episodes.csv files. No Malmo needed.

Metrics (per run, over non-overlapping windows of --window episodes):
  1. kills per window            = episodes with outcome "killed"
                                   (in multi-kill envs "killed" = all prey dead)
  2. % episodes ended by hazard  = outcome in HAZARD_OUTCOMES (fell, stuck_in_pit, ...)
Experiment 1:
  env steps to threshold         = cumulative env steps at the first window whose
                                   kill rate >= --threshold (None if never reached)

A run is one label; several CSV files for a label (resumed training) are
concatenated in filename (= timestamp) order, with env steps accumulated.

Usage:
  python evaluation/compare_sample_efficiency.py \\
      --run ppo='logs/hunting_ppo_*_episodes.csv' \\
      --run dreamer='logs/hunting_dreamer_*_episodes.csv' \\
      --window 20 --threshold 0.5 [--plot out.png] [--csv out.csv]
"""
import argparse
import csv
import glob
import sys

HAZARD_OUTCOMES = {"fell", "stuck_in_pit", "drowned", "lava", "burned", "suffocated", "hazard"}


def kills_per_episode(episodes_csv):
    """{episode: kills} from the sibling *_trajectories.csv, which logs every kill
    (outcome 'kill' mid-episode, 'killed' on the last one). The episode CSV only
    keeps the final outcome, so a 1-2 kill episode that then times out reads as
    0 kills there. Returns None if there is no trajectory file."""
    import os
    traj = episodes_csv.replace("_episodes.csv", "_trajectories.csv")
    if not os.path.exists(traj):
        return None
    out = {}
    with open(traj, newline="") as f:
        rows = csv.reader(l for l in f if not l.startswith("#"))
        hdr = next(rows, None)
        if not hdr or "outcome" not in hdr:
            return None
        ie, io = hdr.index("episode"), hdr.index("outcome")
        for r in rows:
            if len(r) > io and r[io] in ("kill", "killed"):
                out[r[ie]] = out.get(r[ie], 0) + 1
    return out


def load_run(pattern):
    """-> list of dict(outcome, steps, reward, env_steps, kills) in training order."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("no files match: {0}".format(pattern))
    eps, total = [], 0
    for fn in files:
        kpe = kills_per_episode(fn)
        with open(fn, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("episode") in (None, "", "episode"):
                    continue
                steps = int(row["steps"])
                total += steps
                eps.append({"outcome": row["outcome"], "steps": steps,
                            "reward": float(row["reward"]), "env_steps": total,
                            "world_seed": row.get("world_seed") or "",
                            "kills": (kpe.get(row["episode"], 0) if kpe is not None
                                      else int(row["outcome"] == "killed"))})
    return eps


def biome_names():
    """seed (str) -> biome name, from HuntingVisionCFG.BIOME_SEEDS; {} if unavailable."""
    try:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from training.configs.hunting_vision_cfg import BIOME_SEEDS
        return {str(v): k for k, v in BIOME_SEEDS.items()}
    except Exception:
        return {}


def by_biome(eps):
    """-> {biome: dict(episodes, kill_pct, hazard_pct, mean_reward)} over all episodes."""
    names = biome_names()
    groups = {}
    for e in eps:
        key = names.get(e["world_seed"], e["world_seed"] or "unknown")
        groups.setdefault(key, []).append(e)
    out = {}
    for key, g in sorted(groups.items()):
        n = float(len(g))
        out[key] = {"episodes": len(g),
                    "kills": sum(e["kills"] for e in g),
                    "kill_pct": 100.0 * sum(e["kills"] > 0 for e in g) / n,
                    "hazard_pct": 100.0 * sum(e["outcome"] in HAZARD_OUTCOMES for e in g) / n,
                    "mean_reward": sum(e["reward"] for e in g) / n}
    return out


def windows(eps, size):
    """Non-overlapping windows; the trailing partial window is dropped."""
    out = []
    for i in range(0, len(eps) - size + 1, size):
        w = eps[i:i + size]
        out.append({
            "end_episode": i + size,
            "env_steps": w[-1]["env_steps"],
            "kills": sum(e["kills"] for e in w),
            "hazard_pct": 100.0 * sum(e["outcome"] in HAZARD_OUTCOMES for e in w) / size,
            "mean_reward": sum(e["reward"] for e in w) / size,
            "mean_steps": sum(e["steps"] for e in w) / size,
        })
    return out


def steps_to_threshold(wins, size, thr):
    for w in wins:
        if w["kills"] / float(size) >= thr:
            return w["env_steps"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=GLOB")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5, help="kill rate per window (0-1)")
    ap.add_argument("--plot", help="save kills-per-window vs env steps (black & white) to PNG")
    ap.add_argument("--csv", help="save per-window metrics to CSV")
    ap.add_argument("--by-biome", action="store_true",
                    help="also print kill / hazard rates per biome (needs the world_seed column)")
    a = ap.parse_args()

    runs = {}
    for spec in a.run:
        label, _, pat = spec.partition("=")
        if not pat:
            raise SystemExit("--run needs LABEL=GLOB, got: " + spec)
        runs[label] = load_run(pat)

    print("window = {0} episodes; threshold = {1:.0%} kills/window".format(a.window, a.threshold))
    print("{0:<12}{1:>9}{2:>11}{3:>14}{4:>14}{5:>16}".format(
        "run", "episodes", "env steps", "kills (last)", "hazard % (last)", "steps->thresh"))
    print("(kills = all kills from the trajectory log; threshold compares kills per episode)")
    all_w = {}
    for label, eps in runs.items():
        w = windows(eps, a.window)
        all_w[label] = w
        s2t = steps_to_threshold(w, a.window, a.threshold)
        last = w[-1] if w else None
        print("{0:<12}{1:>9}{2:>11}{3:>14}{4:>14}{5:>16}".format(
            label, len(eps), eps[-1]["env_steps"],
            "{0}/{1}".format(last["kills"], a.window) if last else "-",
            "{0:.0f}".format(last["hazard_pct"]) if last else "-",
            s2t if s2t is not None else "not reached"))
        if not w:
            print("   (fewer than {0} episodes: no complete window)".format(a.window))

    if a.by_biome:
        print("\nper biome (all episodes):")
        print("{0:<12}{1:<10}{2:>9}{3:>9}{4:>11}{5:>13}".format(
            "run", "biome", "episodes", "kills", "hazard %", "mean reward"))
        for label, eps in runs.items():
            for biome, s in by_biome(eps).items():
                print("{0:<12}{1:<10}{2:>9}{3:>9}{4:>11.0f}{5:>13.2f}".format(
                    label, biome, s["episodes"], s["kills"], s["hazard_pct"], s["mean_reward"]))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["run", "end_episode", "env_steps", "kills", "hazard_pct", "mean_reward", "mean_steps"])
            for label, ws in all_w.items():
                for w in ws:
                    wr.writerow([label, w["end_episode"], w["env_steps"], w["kills"],
                                 round(w["hazard_pct"], 1), round(w["mean_reward"], 3), round(w["mean_steps"], 1)])
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "d")]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for i, (label, ws) in enumerate(all_w.items()):
            ls, mk = styles[i % len(styles)]
            ax.plot([w["env_steps"] for w in ws], [w["kills"] for w in ws],
                    color="black", linestyle=ls, marker=mk, markersize=3, label=label)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("kills per {0} episodes".format(a.window))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=200)
        print("wrote", a.plot)


if __name__ == "__main__":
    sys.exit(main())
