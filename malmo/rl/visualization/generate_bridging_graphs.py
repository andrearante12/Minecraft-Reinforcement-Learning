"""
visualization/generate_bridging_graphs.py
------------------------------------------
Produces two sets of publication-ready graphs for the bridging task:

  baseline/      — straight 5-block gap (runs up to commit ea1fecd, 2026-04-23 22:21)
  diagonal/      — diagonal gap finetuning (runs after ea1fecd, 2026-04-23 onwards)

Each set contains:
  success_rate_curve.png  — rolling success rate over training
  reward_curve.png        — rolling mean reward over training
  steps_per_episode.png   — rolling episode length over training
  blocks_used.png         — rolling blocks placed on successful episodes
  outcome_distribution.png — outcome breakdown for the best run in that set

Usage (from repo root):
    conda activate train_env
    python Malmo/rl/visualization/generate_bridging_graphs.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
LOG_DIR    = os.path.join(REPO_ROOT, "Malmo", "rl", "logs")

BASELINE_DIR = os.path.join(SCRIPT_DIR, "bridging_graphs", "baseline")
DIAGONAL_DIR = os.path.join(SCRIPT_DIR, "bridging_graphs", "diagonal")
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(DIAGONAL_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":    150,
})

OUTCOME_COLORS = {
    "landed":        "#55A868",
    "fell":          "#C44E52",
    "near_miss":     "#DD8452",
    "mission_ended": "#8172B2",
    "timeout":       "#937860",
}

# ── Run file lists ─────────────────────────────────────────────────────────────
# Baseline: straight 5-block gap, up to and including commit ea1fecd (2026-04-23 22:21).
# Filtered to >= 50 episodes and <= 80% mission_ended to exclude pure Malmo glitches.
BASELINE_FILES = [
    ("sb3_ppo_bridging_20260407_183632_episodes.csv", "105 eps, 0%"),
    ("sb3_ppo_bridging_20260407_190914_episodes.csv", "201 eps, 0%"),
    ("sb3_ppo_bridging_20260408_133745_episodes.csv", "295 eps, 0%"),
    ("sb3_ppo_bridging_20260421_212229_episodes.csv", "305 eps, 0%"),
    ("sb3_ppo_bridging_20260422_081114_episodes.csv", "1460 eps, 0%"),
    ("sb3_ppo_bridging_20260422_173046_episodes.csv", "1563 eps, 9%"),
    ("sb3_ppo_bridging_20260423_184727_episodes.csv", "219 eps, 32%"),
]
BASELINE_BEST = "sb3_ppo_bridging_20260423_184727_episodes.csv"

# Diagonal: finetuning from baseline checkpoint onto diagonal gap (after ea1fecd).
# April 30 001446 excluded from learning curve (drops to 80%, likely a fresh
# checkpoint reload restarting the curve) but included in outcome distribution.
DIAGONAL_FILES = [
    ("sb3_ppo_bridging_20260423_232836_episodes.csv", "771 eps, 21%"),
    ("sb3_ppo_bridging_20260424_134422_episodes.csv", "875 eps, 70%"),
    ("sb3_ppo_bridging_20260425_035057_episodes.csv", "659 eps, 96%"),
]
DIAGONAL_BEST = "sb3_ppo_bridging_20260425_035057_episodes.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────
def rolling(series, window, min_periods=1):
    return series.rolling(window, min_periods=min_periods).mean()


def load_runs(file_list):
    """Load and concatenate a list of (filename, label) tuples, renumbering globally."""
    dfs = []
    for fname, _ in file_list:
        path = os.path.join(LOG_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        dfs.append(pd.read_csv(path))
    combined = pd.concat(dfs, ignore_index=True)
    combined["global_episode"] = range(1, len(combined) + 1)
    combined["landed"] = (combined["outcome"] == "landed").astype(int)
    return combined


def load_single(fname):
    path = os.path.join(LOG_DIR, fname)
    df = pd.read_csv(path)
    df["global_episode"] = range(1, len(df) + 1)
    df["landed"] = (df["outcome"] == "landed").astype(int)
    return df


# ── Plotting functions ────────────────────────────────────────────────────────
def plot_success_rate(df, out_path, title):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = 100

    ax.scatter(df["global_episode"], df["landed"] * 100,
               s=1.5, alpha=0.08, color="#55A868", linewidths=0)

    roll = rolling(df["landed"], window) * 100
    ax.plot(df["global_episode"], roll, color="#222222", linewidth=2,
            label=f"{window}-episode rolling mean", zorder=5)

    final_sr = roll.iloc[-1]
    ax.annotate(
        f"Final: {final_sr:.0f}%",
        xy=(df["global_episode"].iloc[-1], final_sr),
        xytext=(-90, 12), textcoords="offset points",
        fontsize=10, color="#222222",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
    )

    ax.set_xlabel("Episode (cumulative across training runs)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(title)
    ax.set_ylim(-5, 105)
    ax.axhline(0,   color="#cccccc", linewidth=0.8, linestyle="--")
    ax.axhline(100, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_reward_curve(df, out_path, title):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = 100

    ax.scatter(df["global_episode"], df["reward"],
               s=1.5, alpha=0.08, color="#4C72B0", linewidths=0)

    roll = rolling(df["reward"], window)
    ax.plot(df["global_episode"], roll, color="#222222", linewidth=2,
            label=f"{window}-episode rolling mean", zorder=5)

    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode (cumulative across training runs)")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_steps_per_episode(df, out_path, title):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = 100

    ax.scatter(df["global_episode"], df["steps"],
               s=1.5, alpha=0.08, color="#937860", linewidths=0)

    roll = rolling(df["steps"], window)
    ax.plot(df["global_episode"], roll, color="#222222", linewidth=2,
            label=f"{window}-episode rolling mean", zorder=5)

    ax.set_xlabel("Episode (cumulative across training runs)")
    ax.set_ylabel("Steps")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_blocks_used(df, out_path, title):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = 100

    success = df[df["landed"] == 1].copy()
    if success.empty:
        print(f"  no successful episodes for blocks_used — skipping {out_path}")
        plt.close(fig)
        return

    ax.scatter(success["global_episode"], success["blocks_placed"],
               s=2, alpha=0.12, color="#DD8452", linewidths=0)

    roll = rolling(success["blocks_placed"].reset_index(drop=True), window)
    ax.plot(success["global_episode"].values, roll.values,
            color="#222222", linewidth=2,
            label=f"{window}-episode rolling mean (successes only)", zorder=5)

    ax.axhline(5, color="#55A868", linewidth=1.2, linestyle="--",
               label="Gap size (5 blocks minimum)")

    ax.set_xlabel("Episode (cumulative across training runs)")
    ax.set_ylabel("Blocks Placed")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_outcome_distribution(df, out_path, suptitle):
    outcome_order = ["landed", "fell", "near_miss", "timeout", "mission_ended"]
    counts = df["outcome"].value_counts()
    total  = len(df)

    present = [o for o in outcome_order if o in counts.index]
    values  = [counts[o] for o in present]
    pcts    = [v / total * 100 for v in values]
    colors  = [OUTCOME_COLORS[o] for o in present]
    labels  = [o.replace("_", " ").title() for o in present]

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(present))
    bar_width = 0.5

    ax_abs.bar(x, values, bar_width, color=colors)
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(labels, rotation=15, ha="right")
    ax_abs.set_ylabel("Episode Count")
    ax_abs.set_title(f"Outcome Counts  (n={total})")

    ax_pct.bar(x, pcts, bar_width, color=colors)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(labels, rotation=15, ha="right")
    ax_pct.set_ylabel("Percentage (%)")
    ax_pct.set_ylim(0, 110)
    ax_pct.set_title("Outcome Distribution (%)")

    for ax, vals in [(ax_abs, values), (ax_pct, pcts)]:
        for xi, v in zip(x, vals):
            fmt = f"{v:.0f}" if ax is ax_abs else f"{v:.1f}%"
            ax.text(xi, v + (max(vals) * 0.02), fmt,
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(label, lc_df, best_df):
    n = len(best_df)
    sr = best_df["landed"].mean() * 100
    srows = best_df[best_df["landed"] == 1]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Learning curve episodes : {len(lc_df):,}")
    print(f"  Best run episodes       : {n}")
    print(f"  Best run success rate   : {sr:.1f}%")
    if not srows.empty:
        print(f"  Mean steps (success)    : {srows['steps'].mean():.1f}")
        print(f"  Mean blocks (success)   : {srows['blocks_placed'].mean():.1f}")
    print(f"  Mean reward             : {best_df['reward'].mean():.1f}")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("=== BASELINE (straight 5-block gap) ===")
    base_lc   = load_runs(BASELINE_FILES)
    base_best = load_single(BASELINE_BEST)
    print(f"  Learning curve: {len(base_lc):,} episodes across {len(BASELINE_FILES)} runs")

    plot_success_rate(base_lc, os.path.join(BASELINE_DIR, "success_rate_curve.png"),
                      "Baseline Bridging — Success Rate over Training\n(Straight 5-Block Gap)")
    plot_reward_curve(base_lc, os.path.join(BASELINE_DIR, "reward_curve.png"),
                      "Baseline Bridging — Episode Reward over Training\n(Straight 5-Block Gap)")
    plot_steps_per_episode(base_lc, os.path.join(BASELINE_DIR, "steps_per_episode.png"),
                           "Baseline Bridging — Episode Length over Training\n(Straight 5-Block Gap)")
    plot_blocks_used(base_lc, os.path.join(BASELINE_DIR, "blocks_used.png"),
                     "Baseline Bridging — Blocks Used per Successful Episode\n(Straight 5-Block Gap)")
    plot_outcome_distribution(base_best,
                              os.path.join(BASELINE_DIR, "outcome_distribution.png"),
                              f"Baseline Bridging — Outcome Distribution (Best Run, {len(base_best)} episodes)\n(Straight 5-Block Gap)")
    print_summary("BASELINE", base_lc, base_best)

    # ── Diagonal finetuning ───────────────────────────────────────────────────
    print("=== DIAGONAL FINETUNING ===")
    diag_lc   = load_runs(DIAGONAL_FILES)
    diag_best = load_single(DIAGONAL_BEST)
    print(f"  Learning curve: {len(diag_lc):,} episodes across {len(DIAGONAL_FILES)} runs")

    plot_success_rate(diag_lc, os.path.join(DIAGONAL_DIR, "success_rate_curve.png"),
                      "Diagonal Bridging — Success Rate over Finetuning\n(Diagonal 5-Block Gap, X+4 Offset)")
    plot_reward_curve(diag_lc, os.path.join(DIAGONAL_DIR, "reward_curve.png"),
                      "Diagonal Bridging — Episode Reward over Finetuning\n(Diagonal 5-Block Gap, X+4 Offset)")
    plot_steps_per_episode(diag_lc, os.path.join(DIAGONAL_DIR, "steps_per_episode.png"),
                           "Diagonal Bridging — Episode Length over Finetuning\n(Diagonal 5-Block Gap, X+4 Offset)")
    plot_blocks_used(diag_lc, os.path.join(DIAGONAL_DIR, "blocks_used.png"),
                     "Diagonal Bridging — Blocks Used per Successful Episode\n(Diagonal 5-Block Gap, X+4 Offset)")
    plot_outcome_distribution(diag_best,
                              os.path.join(DIAGONAL_DIR, "outcome_distribution.png"),
                              f"Diagonal Bridging — Outcome Distribution (Best Run, {len(diag_best)} episodes)\n(Diagonal 5-Block Gap, X+4 Offset)")
    print_summary("DIAGONAL FINETUNING", diag_lc, diag_best)
