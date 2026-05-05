"""
Quick training plotter — plots BC loss curve and PPO reward/success/steps graphs.
Usage:
    conda activate train_env
    python Malmo/rl/visualization/plot_training.py
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "logs")
OUT_DIR    = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── BC loss curve ─────────────────────────────────────────────────────────────
bc_updates = sorted(glob.glob(os.path.join(LOG_DIR, "*bc*updates.csv")))
if bc_updates:
    bc_path = bc_updates[-1]
    print("BC updates:", os.path.basename(bc_path))
    bc_df = pd.read_csv(bc_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("BC Training — " + os.path.basename(bc_path).replace("_updates.csv", ""), fontsize=12)

    axes[0].plot(bc_df["update"], bc_df["bc_loss"], color="crimson", linewidth=2)
    axes[0].set_ylabel("BC Loss")
    axes[0].set_title("Loss (lower = better)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bc_df["update"], bc_df["accuracy"] * 100, color="steelblue", linewidth=2)
    axes[1].set_ylabel("Accuracy %")
    axes[1].set_title("Action Prediction Accuracy (higher = better)")
    axes[1].set_xlabel("Update")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "bc_loss.png")
    plt.savefig(out, dpi=150)
    print("Saved to:", out)
    plt.close()

# ── PPO reward/success/steps ──────────────────────────────────────────────────
ppo_csvs = sorted([f for f in glob.glob(os.path.join(LOG_DIR, "*ppo*episodes.csv"))
                   if "sb3" not in os.path.basename(f)])
if not ppo_csvs:
    print("No PPO episode CSVs found — skipping PPO plots.")
else:
    path = ppo_csvs[-1]
    print("PPO episodes:", os.path.basename(path))
    df = pd.read_csv(path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(os.path.basename(path).replace("_episodes.csv", ""), fontsize=13)

    ax = axes[0]
    ax.plot(df["episode"], df["reward"], alpha=0.3, color="steelblue", linewidth=0.8)
    if len(df) >= 10:
        rolling = df["reward"].rolling(window=20, min_periods=1).mean()
        ax.plot(df["episode"], rolling, color="steelblue", linewidth=2, label="20-ep avg")
        ax.legend()
    ax.set_ylabel("Reward")
    ax.set_title("Reward per Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "outcome" in df.columns:
        df["success"] = (df["outcome"] == "landed").astype(float)
        rolling_sr = df["success"].rolling(window=20, min_periods=1).mean() * 100
        ax.plot(df["episode"], rolling_sr, color="green", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Success Rate %")
        ax.set_title("Success Rate (20-ep rolling)")
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    if "steps" in df.columns:
        ax.plot(df["episode"], df["steps"], alpha=0.3, color="orange", linewidth=0.8)
        rolling_steps = df["steps"].rolling(window=20, min_periods=1).mean()
        ax.plot(df["episode"], rolling_steps, color="orange", linewidth=2, label="20-ep avg")
        ax.legend()
        ax.set_ylabel("Steps")
        ax.set_title("Steps per Episode")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "ppo_training.png")
    plt.savefig(out_path, dpi=150)
    print("Saved to:", out_path)
