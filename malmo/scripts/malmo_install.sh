#!/usr/bin/env bash
#
# malmo_install.sh — Linux (Ubuntu 24.04+) counterpart of malmo_install.ps1.
#
# Mirrors the PowerShell installer's step structure:
#   install deps -> create conda envs -> set MALMO_XSD_PATH / PYTHONPATH.
# It is additive and idempotent: re-running skips work already done and only
# appends to ~/.bashrc lines that are not already present.
#
# What it does NOT do (the risky, network-bound parts — see setup_linux.md):
#   * download the prebuilt Malmo 0.37.0 Linux release (MalmoPython.so + the
#     Minecraft/ and Schemas/ trees) — you do that by hand, once.
#   * build MalmoPython from source.
#
# Usage:
#   bash malmo/scripts/malmo_install.sh            # full run (apt needs sudo)
#   bash malmo/scripts/malmo_install.sh --skip-apt # re-run without touching apt/java
#
set -euo pipefail

SKIP_APT=0
for arg in "$@"; do
  case "$arg" in
    --skip-apt) SKIP_APT=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# MALMO_HOME = the malmo/ dir (parent of scripts/), mirroring `Get-Item ..` in the ps1.
MALMO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "==> MALMO_HOME = $MALMO_HOME"

# ---------------------------------------------------------------------------
# 1. System dependencies (apt) + Java 8 selection
# ---------------------------------------------------------------------------
if [[ "$SKIP_APT" -eq 0 ]]; then
  echo "==> Installing system dependencies (needs sudo)..."
  sudo apt-get update
  # openjdk-8: Malmo's Minecraft is Forge 1.11.2 and requires Java 8.
  # libboost-all-dev: general boost tooling (the py3.7 boost-python shim is conda, step 3).
  # ffmpeg / python3-tk: Malmo recording + UI deps.
  sudo apt-get install -y openjdk-8-jdk libboost-all-dev ffmpeg python3-tk

  echo "==> Selecting Java 8 as the active 'java'..."
  JAVA8="$(update-alternatives --list java 2>/dev/null | grep -m1 'java-8' || true)"
  if [[ -n "$JAVA8" ]]; then
    sudo update-alternatives --set java "$JAVA8"
    echo "    java -> $JAVA8"
  else
    echo "    WARNING: could not auto-find a java-8 alternative."
    echo "    Run 'sudo update-alternatives --config java' and pick the openjdk-8 entry manually."
  fi
  java -version || true
else
  echo "==> --skip-apt: skipping apt + java selection."
fi

# ---------------------------------------------------------------------------
# 2. Conda environments
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

create_env_if_missing() {
  local name="$1" file="$2"
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "==> conda env '$name' already exists — skipping."
  else
    echo "==> Creating conda env '$name' from $file ..."
    conda env create -f "$file"
  fi
}

create_env_if_missing "malmo"     "$MALMO_HOME/conda_enviornments/malmo_environment_linux.yml"
create_env_if_missing "train_env" "$MALMO_HOME/conda_enviornments/training_environment_linux.yml"

# ---------------------------------------------------------------------------
# 3. MalmoPython.so (Linux counterpart of "copy MalmoPython.pyd")
# ---------------------------------------------------------------------------
# The prebuilt Ubuntu-18.04 / Python-3.6 MalmoPython.so STATICALLY links boost
# (it is a "_withBoost_" build) and resolves Python symbols from the host
# interpreter, so it needs neither a boost package nor LD_LIBRARY_PATH — only a
# Python 3.6 interpreter (which is why the `malmo` env above is 3.6). Verified:
#   readelf -d malmo/Python_Examples/MalmoPython.so | grep NEEDED   # no boost/python
# Placing the .so is a manual step (network download) — see setup_linux.md.
echo "==> NOTE: place MalmoPython.so in malmo/Python_Examples/ (see setup_linux.md)."

# ---------------------------------------------------------------------------
# 4. Environment variables -> ~/.bashrc (append-only, de-duplicated)
# ---------------------------------------------------------------------------
BASHRC="$HOME/.bashrc"
add_line() {
  local line="$1"
  if ! grep -qxF "$line" "$BASHRC" 2>/dev/null; then
    echo "$line" >> "$BASHRC"
    echo "    + $line"
  fi
}

echo "==> Adding env vars to $BASHRC (skipping any already present)..."
add_line "# --- MalmoRL (added by malmo_install.sh) ---"
# MALMO_XSD_PATH: Malmo validates mission XML against these XSDs at runtime.
add_line "export MALMO_XSD_PATH=\"$MALMO_HOME/Schemas\""
# PYTHONPATH: lets `import envs...` / `import training...` resolve without cwd games.
add_line "export PYTHONPATH=\"$MALMO_HOME/rl:\${PYTHONPATH:-}\""
# NOTE: no LD_LIBRARY_PATH needed — the prebuilt MalmoPython.so static-links boost
# and gets Python symbols from the interpreter (verified via `readelf -d ... NEEDED`).

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

==> Base install complete.

Next (manual — see malmo/docs/framework/setup_linux.md):
  1. Download the Malmo 0.37.0 Linux release (Ubuntu-18.04 / Python-3.6 build) and copy:
       <release>/Minecraft/         -> $MALMO_HOME/Minecraft/
       <release>/Python_Examples/MalmoPython.so -> $MALMO_HOME/Python_Examples/
     (Schemas/ already ships in the repo and matches the 0.37.0 release.)
  2. Open a new shell (or 'source ~/.bashrc'), then verify:
       conda activate malmo
       python -c "import MalmoPython; print(MalmoPython.AgentHost)"
  3. Launch everything with: WorkflowScripts/launch_training.sh
EOF
