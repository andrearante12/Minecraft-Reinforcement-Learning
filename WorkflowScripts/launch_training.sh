#!/usr/bin/env bash
#
# launch_training.sh — Linux counterpart of launch_training.ps1.
#
# Opens a tmux session with 3 panes, one per role (matching the ps1's 3 tabs):
#   1. MC-Client : runs ./launchClient.sh
#   2. EnvServer : activates the `malmo` conda env (you type the env_server.py command)
#   3. Training  : activates the `train_env` conda env (you type the train_sb3.py command)
#
# tmux is used instead of a desktop terminal so this also works over SSH / headless.
#
# Usage:  WorkflowScripts/launch_training.sh
# Detach: Ctrl-b then d     Reattach: tmux attach -t malmorl     Kill: tmux kill-session -t malmorl
#
set -euo pipefail

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Install it with: sudo apt-get install -y tmux" >&2
  exit 1
fi

# projectRoot = parent of this script's dir (mirrors Split-Path -Parent $PSScriptRoot).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="malmorl"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists — attaching. (Kill it with: tmux kill-session -t $SESSION)"
  exec tmux attach -t "$SESSION"
fi

# Make `conda activate` work inside non-interactive tmux panes.
CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"

# Pane 1: Minecraft client (the only pane that auto-runs a command, as in the ps1).
tmux new-session -d -s "$SESSION" -n malmorl \
  "cd '$PROJECT_ROOT/malmo/Minecraft' && ./launchClient.sh; exec bash"

# Pane 2: Environment server (conda malmo).
tmux split-window -t "$SESSION" -v \
  "cd '$PROJECT_ROOT'; [ -f '$CONDA_SH' ] && source '$CONDA_SH'; conda activate malmo; exec bash"

# Pane 3: Training (conda train_env).
tmux split-window -t "$SESSION" -v \
  "cd '$PROJECT_ROOT'; [ -f '$CONDA_SH' ] && source '$CONDA_SH'; conda activate train_env; exec bash"

tmux select-layout -t "$SESSION" even-vertical
tmux attach -t "$SESSION"
