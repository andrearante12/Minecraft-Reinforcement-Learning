"""Summarise the current hunting_vision training era into REPORT.md (+ reward_curve.png).

Run:  python malmo/rl/utils/training_report.py [--out DIR] [--first-episode 1321]

Merges every <run>_episodes.csv from the first run of the era, de-duplicating
episode numbers by keeping the row from the latest run file (restarts replay a
few episode numbers). Phases mark when the reward / world changed.
"""
import argparse
import collections
import csv
import glob
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")
CKPT = os.path.join(HERE, "..", "checkpoints")

# (first episode, label) — edit when the reward or world changes.
PHASES = [
    (1321, "pitch shaping + extreme-pitch penalty"),
    (1326, "+ pit penalty / early end (pond present)"),
    (1376, "+ pond filled with grass"),
    (1601, "multi-world (5 seeds) + random pits/ponds, pond unfilled"),
]
KILL = "killed"
PIT = "stuck_in_pit"


def load_episodes(first_run, first_ep):
    rows = {}
    for f in sorted(glob.glob(os.path.join(LOGS, "hunting_vision_dreamer_video_*_episodes.csv"))):
        if os.path.basename(f).split("_dreamer_video_")[1] < first_run:
            continue
        for r in csv.DictReader(open(f)):
            e = int(r["episode"])
            if e >= first_ep:
                rows[e] = dict(ep=e, reward=float(r["reward"]), steps=int(r["steps"]),
                               outcome=r["outcome"], t=r["timestamp"])
    return [rows[e] for e in sorted(rows)]


def phase_of(ep):
    label = PHASES[0][1]
    for start, name in PHASES:
        if ep >= start:
            label = name
    return label


def summarise(eps):
    n = len(eps)
    if not n:
        return dict(n=0)
    c = collections.Counter(e["outcome"] for e in eps)
    return dict(
        n=n, kill=c[KILL] / n, pit=c[PIT] / n, timeout=c["timeout"] / n,
        early_pit=sum(1 for e in eps if e["outcome"] == PIT and e["steps"] < 100) / n,
        mean=sum(e["reward"] for e in eps) / n,
        pos=sum(1 for e in eps if e["reward"] > 0) / n,
        good=sum(1 for e in eps if e["reward"] >= 150) / n,
        counts=dict(c))


def row(name, s):
    if not s["n"]:
        return "| {0} | 0 | | | | | | |".format(name)
    return "| {0} | {1} | {2:.0%} | {3:.0%} | {4:.0%} | {5:.0%} | {6:.0f} | {7:.0%} |".format(
        name, s["n"], s["kill"], s["pit"], s["early_pit"], s["timeout"], s["mean"], s["good"])


HEAD = ("| {0} | episodes | 3-kill finish | pit end | early pit (<100 steps) | "
        "timeout | mean reward | reward >= 150 |\n|---|---|---|---|---|---|---|---|")


def chart(eps, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {KILL: "#2a9d3a", PIT: "#c0392b", "timeout": "#7f8c8d"}
    for o, col in colors.items():
        pts = [(e["ep"], e["reward"]) for e in eps if e["outcome"] == o]
        if pts:
            ax.scatter(*zip(*pts), s=14, c=col, label=o)
    w = 10
    if len(eps) >= w:
        xs = [eps[i]["ep"] for i in range(w - 1, len(eps))]
        ys = [sum(e["reward"] for e in eps[i - w + 1:i + 1]) / w for i in range(w - 1, len(eps))]
        ax.plot(xs, ys, c="black", lw=1.5, label="{0}-ep mean".format(w))
    for start, _ in PHASES[1:]:
        if eps[0]["ep"] < start <= eps[-1]["ep"]:
            ax.axvline(start - 0.5, c="#2980b9", ls="--", lw=1)
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "recordings", "overnight"))
    ap.add_argument("--first-episode", type=int, default=PHASES[0][0])
    ap.add_argument("--first-run", default="20261011_004420")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    eps = load_episodes(a.first_run, a.first_episode)
    if not eps:
        print("no episodes found")
        return
    L = []
    L.append("# Hunting training report (auto-generated)\n")
    L.append("Generated {0}. Episodes {1}-{2} ({3} total), clock {4} -> {5}.\n".format(
        time.strftime("%Y-%m-%d %H:%M"), eps[0]["ep"], eps[-1]["ep"], len(eps),
        eps[0]["t"], eps[-1]["t"]))

    L.append("## Headline\n")
    last = eps[-25:]
    sl, sa = summarise(last), summarise(eps)
    L.append("- Last {0} episodes: {1:.0%} finish all 3 cows, {2:.0%} end in a pit, mean return {3:.0f}.".format(
        sl["n"], sl["kill"], sl["pit"], sl["mean"]))
    L.append("- Whole era: {0:.0%} 3-kill finishes, {1:.0%} pit endings, mean return {2:.0f}.".format(
        sa["kill"], sa["pit"], sa["mean"]))
    best = max(eps, key=lambda e: e["reward"])
    L.append("- Best episode: #{0}, return {1:.0f} ({2}, {3} steps).\n".format(
        best["ep"], best["reward"], best["outcome"], best["steps"]))

    L.append("## By phase\n")
    L.append(HEAD.format("phase"))
    by = collections.OrderedDict()
    for e in eps:
        by.setdefault(phase_of(e["ep"]), []).append(e)
    for name, es in by.items():
        L.append(row(name, summarise(es)))
    L.append("")

    L.append("## By 25-episode window\n")
    L.append(HEAD.format("episodes"))
    for i in range(0, len(eps), 25):
        w = eps[i:i + 25]
        L.append(row("{0}-{1}".format(w[0]["ep"], w[-1]["ep"]), summarise(w)))
    L.append("")

    if chart(eps, os.path.join(out, "reward_curve.png")):
        L.append("![return per episode](reward_curve.png)\n")

    L.append("## Top episodes\n")
    L.append("| episode | return | steps | outcome |\n|---|---|---|---|")
    for e in sorted(eps, key=lambda e: -e["reward"])[:8]:
        L.append("| {0} | {1:.0f} | {2} | {3} |".format(e["ep"], e["reward"], e["steps"], e["outcome"]))
    L.append("")

    clips = sorted(glob.glob(os.path.join(out, "*.mp4")))
    L.append("## Recorded clips ({0})\n".format(len(clips)))
    for c in clips:
        L.append("- `{0}` ({1:.1f} MB)".format(os.path.basename(c), os.path.getsize(c) / 1e6))
    L.append("")

    cks = sorted(glob.glob(os.path.join(CKPT, "dreamer_video_hunting_vision_dreamer_video_ep*.pt")),
                 key=os.path.getmtime)[-3:]
    L.append("## Latest checkpoints\n")
    for c in cks:
        L.append("- `{0}` ({1})".format(os.path.basename(c), time.strftime(
            "%H:%M", time.localtime(os.path.getmtime(c)))))
    L.append("")
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L))
    print("wrote", os.path.join(out, "REPORT.md"))


if __name__ == "__main__":
    main()
