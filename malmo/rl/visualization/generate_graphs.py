"""
visualization/generate_graphs.py
---------------------------------
Reads all curriculum_ppo episode CSVs from Malmo/rl/logs/ and produces
4 publication-ready PNG graphs in Malmo/rl/visualization/graphs/.

Usage (from repo root):
    conda activate train_env
    python Malmo/rl/visualization/generate_graphs.py

Output files:
    graphs/reward_curve.png          — rolling-mean reward vs episode
    graphs/success_rate.png          — rolling success rate per environment
    graphs/outcome_distribution.png  — fell / landed bar chart per environment
    graphs/steps_per_episode.png     — episode length over training
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
LOG_DIR    = os.path.join(REPO_ROOT, "Malmo", "rl", "logs")
OUT_DIR    = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
ENV_COLORS = {
    "one_block_gap":   "#4C72B0",
    "simple_jump":     "#55A868",
    "three_block_gap": "#C44E52",
    "vertical_small":  "#DD8452",
    "diagonal_small":  "#8172B2",
    "diagonal_medium": "#937860",
}
ENV_LABELS = {
    "one_block_gap":   "One-Block Gap",
    "simple_jump":     "Simple Jump",
    "three_block_gap": "Three-Block Gap",
    "vertical_small":  "Vertical Small",
    "diagonal_small":  "Diagonal Small",
    "diagonal_medium": "Diagonal Medium",
}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":    150,
})


# ── Load data ─────────────────────────────────────────────────────────────────
def load_logs():
    pattern = os.path.join(LOG_DIR, "curriculum_ppo_*_episodes.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No curriculum episode CSVs found in {LOG_DIR}")
    dfs = [pd.read_csv(f) for f in files]
    df = (pd.concat(dfs)
            .drop_duplicates("episode")
            .sort_values("episode")
            .reset_index(drop=True))
    df["landed"] = (df["outcome"] == "landed").astype(int)
    return df


def rolling(series, window, min_periods=1):
    return series.rolling(window, min_periods=min_periods).mean()


def env_spans(df):
    """Return list of (env, start_ep, end_ep) for background shading."""
    spans = []
    cur_env, cur_start = df.iloc[0]["env"], df.iloc[0]["episode"]
    for _, row in df.iterrows():
        if row["env"] != cur_env:
            spans.append((cur_env, cur_start, row["episode"]))
            cur_env, cur_start = row["env"], row["episode"]
    spans.append((cur_env, cur_start, df.iloc[-1]["episode"]))
    return spans


def shade_envs(ax, spans, alpha=0.08):
    for env, start, end in spans:
        color = ENV_COLORS.get(env, "#888888")
        ax.axvspan(start, end, alpha=alpha, color=color, linewidth=0)


# ── Graph 1: Reward curve ─────────────────────────────────────────────────────
def plot_reward_curve(df, out_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    spans = env_spans(df)
    shade_envs(ax, spans)

    # Raw scatter (tiny, low opacity)
    for env, color in ENV_COLORS.items():
        sub = df[df["env"] == env]
        ax.scatter(sub["episode"], sub["reward"], s=1, alpha=0.12,
                   color=color, linewidths=0)

    # Rolling mean per contiguous env block (preserves curriculum ordering)
    window = 200
    roll = rolling(df["reward"], window)
    ax.plot(df["episode"], roll, color="#222222", linewidth=2,
            label=f"{window}-ep rolling mean", zorder=5)

    # Legend for environments
    patches = [mpatches.Patch(color=c, label=ENV_LABELS.get(e, e))
               for e, c in ENV_COLORS.items() if e in df["env"].values]
    first_legend = ax.legend(handles=patches, loc="upper left",
                             title="Environment", framealpha=0.8)
    ax.add_artist(first_legend)
    ax.legend(loc="lower right", framealpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward over Curriculum Training")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Graph 2: Success rate ─────────────────────────────────────────────────────
def plot_success_rate(df, out_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    spans = env_spans(df)
    shade_envs(ax, spans, alpha=0.10)

    window = 300
    roll = rolling(df["landed"], window) * 100
    ax.plot(df["episode"], roll, color="#222222", linewidth=2,
            label=f"{window}-ep rolling rate", zorder=5)

    # Per-env rolling lines (thinner, colored)
    for env, color in ENV_COLORS.items():
        mask = df["env"] == env
        if not mask.any():
            continue
        sub = df[mask].copy()
        r = rolling(sub["landed"], min(200, len(sub))) * 100
        ax.plot(sub["episode"], r, color=color, linewidth=1.2,
                alpha=0.7, label=ENV_LABELS.get(env, env))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Rolling Success Rate (\"Landed\") over Curriculum Training")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="upper left", framealpha=0.8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Graph 3: Outcome distribution ────────────────────────────────────────────
def plot_outcome_distribution(df, out_path):
    outcome_order = ["landed", "fell", "near_miss", "mission_ended"]
    outcome_colors = {
        "landed":        "#55A868",
        "fell":          "#C44E52",
        "near_miss":     "#DD8452",
        "mission_ended": "#8172B2",
    }

    # Build per-env counts
    env_order = [e for e in ENV_COLORS if e in df["env"].values]
    counts = (df.groupby(["env", "outcome"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=env_order))
    # Ensure all outcome columns exist
    for o in outcome_order:
        if o not in counts.columns:
            counts[o] = 0
    counts = counts[outcome_order]

    # Normalise to percentages
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(13, 5))

    xlabels = [ENV_LABELS.get(e, e) for e in env_order]
    x = np.arange(len(xlabels))
    bar_width = 0.55

    for ax, data, title in [
        (ax_abs, counts, "Episode Counts per Environment"),
        (ax_pct, pct,    "Outcome Distribution (%) per Environment"),
    ]:
        bottom = np.zeros(len(env_order))
        for outcome in outcome_order:
            vals = data[outcome].values.astype(float)
            ax.bar(x, vals, bar_width, bottom=bottom,
                   color=outcome_colors[outcome],
                   label=outcome.replace("_", " ").title())
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.85)

    ax_abs.set_ylabel("Episodes")
    ax_pct.set_ylabel("Percentage (%)")
    ax_pct.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Graph 4: Steps per episode ───────────────────────────────────────────────
def plot_steps_per_episode(df, out_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    spans = env_spans(df)
    shade_envs(ax, spans)

    ax.scatter(df["episode"], df["steps"], s=1, alpha=0.12,
               color="#555555", linewidths=0)

    window = 200
    roll = rolling(df["steps"], window)
    ax.plot(df["episode"], roll, color="#222222", linewidth=2,
            label=f"{window}-ep rolling mean", zorder=5)

    patches = [mpatches.Patch(color=c, label=ENV_LABELS.get(e, e))
               for e, c in ENV_COLORS.items() if e in df["env"].values]
    first_legend = ax.legend(handles=patches, loc="upper right",
                             title="Environment", framealpha=0.8)
    ax.add_artist(first_legend)
    ax.legend(loc="upper left", framealpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length over Curriculum Training")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Graph 5: Mixed-maintenance progress ──────────────────────────────────────
def plot_mixed_maintenance_progress(df, out_path, start_episode=11601):
    """Per-env rolling success rate since the mixed-maintenance checkpoint."""
    data = df[df["episode"] >= start_episode].copy()
    if data.empty:
        print("  no mixed-maintenance data to plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    window = 100

    for env, color in ENV_COLORS.items():
        mask = data["env"] == env
        if not mask.any():
            continue
        sub = data[mask].sort_values("episode")
        r = rolling(sub["landed"], min(window, len(sub))) * 100
        ax.plot(sub["episode"], r, color=color, linewidth=1.8,
                label=ENV_LABELS.get(env, env))

        # Annotate final success rate
        final_sr = r.iloc[-1]
        ax.annotate(
            f"{final_sr:.0f}%",
            xy=(sub["episode"].iloc[-1], final_sr),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=color, va="center",
        )

    ax.axvline(start_episode, color="#aaaaaa", linewidth=1.0,
               linestyle="--", label=f"Checkpoint ep {start_episode:,}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Per-Environment Success Rate — Mixed Maintenance Curriculum")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="upper left", framealpha=0.85, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n" + "=" * 70)
    print(f"  CURRICULUM TRAINING SUMMARY  —  {len(df):,} total episodes")
    print("=" * 70)
    summary = (df.groupby("env")
                 .agg(
                     episodes=("episode", "count"),
                     success_rate=("landed", "mean"),
                     mean_reward=("reward", "mean"),
                     mean_steps=("steps", "mean"),
                 )
                 .reindex([e for e in ENV_COLORS if e in df["env"].values])
                 .round(3))
    summary["success_rate"] = (summary["success_rate"] * 100).round(1).astype(str) + "%"
    summary.index = [ENV_LABELS.get(e, e) for e in summary.index]
    summary.columns = ["Episodes", "Success Rate", "Mean Reward", "Mean Steps"]
    print(summary.to_string())
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading logs...")
    df = load_logs()
    print(f"  Loaded {len(df):,} episodes across {df['env'].nunique()} environments.\n")

    print("Generating graphs:")
    plot_reward_curve(df,           os.path.join(OUT_DIR, "reward_curve.png"))
    plot_success_rate(df,           os.path.join(OUT_DIR, "success_rate.png"))
    plot_outcome_distribution(df,   os.path.join(OUT_DIR, "outcome_distribution.png"))
    plot_steps_per_episode(df,      os.path.join(OUT_DIR, "steps_per_episode.png"))
    plot_mixed_maintenance_progress(df, os.path.join(OUT_DIR, "mixed_maintenance_progress.png"))

    print_summary(df)
