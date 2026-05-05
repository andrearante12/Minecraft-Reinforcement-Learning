"""
Quick training plotter — reads the most recent PPO episodes CSV and saves a reward graph.
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

# Pick the most recent ppo episodes CSV
csvs = sorted(glob.glob(os.path.join(LOG_DIR, "*ppo*episodes.csv")))
if not csvs:
    raise FileNotFoundError("No PPO episode CSVs found in " + LOG_DIR)

path = csvs[-1]
print("Reading:", os.path.basename(path))

df = pd.read_csv(path)
print("  Episodes loaded:", len(df))
print("  Columns:", list(df.columns))

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(os.path.basename(path).replace("_episodes.csv", ""), fontsize=13)

# ── Reward curve ──────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(df["episode"], df["reward"], alpha=0.3, color="steelblue", linewidth=0.8)
if len(df) >= 10:
    rolling = df["reward"].rolling(window=20, min_periods=1).mean()
    ax.plot(df["episode"], rolling, color="steelblue", linewidth=2, label="20-ep avg")
    ax.legend()
ax.set_ylabel("Reward")
ax.set_title("Reward per Episode")
ax.grid(True, alpha=0.3)

# ── Success rate ──────────────────────────────────────────────────────────────
ax = axes[1]
if "outcome" in df.columns:
    df["success"] = (df["outcome"] == "landed").astype(float)
    rolling_sr = df["success"].rolling(window=20, min_periods=1).mean() * 100
    ax.plot(df["episode"], rolling_sr, color="green", linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate %")
    ax.set_title("Success Rate (20-ep rolling)")
    ax.grid(True, alpha=0.3)

# ── Steps per episode ─────────────────────────────────────────────────────────
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
