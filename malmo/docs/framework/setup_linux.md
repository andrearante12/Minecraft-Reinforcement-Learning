# Setup & Installation (Linux)

Linux (Ubuntu 24.04+) counterpart of [Setup & Installation (Windows)](./setup.md).
The RL framework code is platform-agnostic; everything Linux-specific is in this doc.

> Most of the steps below are automated by `malmo/scripts/malmo_install.sh`. Read this
> doc to understand what it does and to handle the one manual part it can't (downloading
> the Malmo release binary).

## Table of Contents

- [Quick path (script)](#quick-path-script)
- [Conda Environment Setup](#conda-environment-setup)
- [System Dependencies & Java 8](#system-dependencies--java-8)
- [Obtaining MalmoPython + the Minecraft tree](#obtaining-malmopython--the-minecraft-tree)
- [Environment Variables](#environment-variables)
- [Verify the Installation](#verify-the-installation)
- [Known Issues](#known-issues)

---

## Quick path (script)

```bash
bash malmo/scripts/malmo_install.sh        # apt deps, Java 8, conda envs, ~/.bashrc exports
# then do the manual "Obtaining MalmoPython" step below, then:
source ~/.bashrc
```

The rest of this doc explains each step.

---

## Conda Environment Setup

Two environments, same split as Windows (`malmo` runs Minecraft + the env server;
`train_env` runs PyTorch training on Python 3.10). Use the Linux env files:

> **The `malmo` env is Python 3.6 on Linux** (the Windows side is 3.7). The only prebuilt
> Linux `MalmoPython.so` for Malmo 0.37.0 is the Ubuntu-18.04 / Python-3.6 build, and a
> prebuilt CPython extension is ABI-locked to its Python minor version. The `malmo` side
> only imports numpy beyond the stdlib, so 3.6 vs 3.7 makes no functional difference.

```bash
conda env create -f malmo/conda_enviornments/malmo_environment_linux.yml
conda env create -f malmo/conda_enviornments/training_environment_linux.yml
```

The Windows `*.yml` files are unchanged and still used on Windows. The `_linux.yml`
files drop the Windows-only conda packages (`vc`, `ucrt`, `vs2015_runtime`, …) and the
build hashes so they solve on Linux.

> Unlike Windows, you do **not** copy a `.pyd` into `site-packages`. On Linux the native
> binding is `MalmoPython.so`, placed in `malmo/Python_Examples/` (see below); the RL code
> already adds that folder to `sys.path`.

---

## System Dependencies & Java 8

```bash
sudo apt-get install -y openjdk-8-jdk libboost-all-dev ffmpeg python3-tk
```

Malmo's Minecraft is **Forge 1.11.2, which requires Java 8**. This machine ships with
Java 21, so select Java 8 as the active `java`:

```bash
sudo update-alternatives --config java   # pick the java-8-openjdk entry
java -version                            # must report 1.8.x
```

(`malmo_install.sh` attempts this automatically via `update-alternatives --set`.)

---

## Obtaining MalmoPython + the Minecraft tree

The native binding and the `Minecraft/` tree are **not** in the repo (the latter is
gitignored). Get them from the prebuilt Malmo **0.37.0** Linux release — the
**`Malmo-0.37.0-Linux-Ubuntu-18.04-64bit_withBoost_Python3.6`** zip (the highest Python a
Linux prebuilt ships; there is no Linux 3.7 build, which is why the `malmo` env is 3.6):

1. Download & unzip the release from https://github.com/microsoft/malmo/releases/tag/0.37.0.
2. Copy into the repo (additive — the Windows `MalmoPython.pyd` stays put; Python picks
   `.so` on Linux automatically):

   ```bash
   cp -r <release>/Minecraft                        malmo/Minecraft
   cp    <release>/Python_Examples/MalmoPython.so   malmo/Python_Examples/
   chmod +x malmo/Minecraft/launchClient.sh
   ```

   `Schemas/` already ships in the repo and matches the 0.37.0 release, so it needs no copy.

3. No boost step is needed. The `_withBoost_` build **statically links** boost and resolves
   Python symbols from the interpreter, so the `.so` needs neither a boost package nor
   `LD_LIBRARY_PATH` — only the Python 3.6 `malmo` env. Confirm with:

   ```bash
   readelf -d malmo/Python_Examples/MalmoPython.so | grep NEEDED   # no libboost / libpython
   ```

**Fallback — build from source.** If the prebuilt `.so` refuses to load (glibc/ABI
mismatch on a very different distro), build Malmo 0.37.0 from source against the conda
Python (`cmake`, `boost`, `swig`, `xsd`, `openjdk-8`, with
`-DPYTHON_EXECUTABLE=$(conda run -n malmo which python)`). Heavier, but robust. Not needed
on Ubuntu 24.04 — the Ubuntu-18.04 prebuilt loads as-is.

---

## Environment Variables

Add to `~/.bashrc` (the Linux equivalent of the Windows registry vars; done automatically
by `malmo_install.sh`):

```bash
export MALMO_XSD_PATH="$HOME/Minecraft-Reinforcement-Learning/malmo/Schemas"
export PYTHONPATH="$HOME/Minecraft-Reinforcement-Learning/malmo/rl:$PYTHONPATH"
```

(No `LD_LIBRARY_PATH` is needed — the prebuilt `.so` static-links boost and gets Python
symbols from the interpreter.)

Then `source ~/.bashrc` and check `echo $MALMO_XSD_PATH`.

> **Use lowercase `malmo/...` in all commands.** Linux is case-sensitive; the directory is
> `malmo`, not `Malmo`. The Windows docs' `Malmo/...` paths only work because Windows is
> case-insensitive.

---

## Verify the Installation

```bash
# 1. Native binding imports
conda activate malmo
python -c "import MalmoPython; print(MalmoPython.AgentHost)"
# on failure: ldd malmo/Python_Examples/MalmoPython.so   # names the missing lib

# 2. Minecraft client launches on Java 8
cd malmo/Minecraft && ./launchClient.sh     # wait for "CLIENT enter state: DORMANT"

# 3. Env server connects (proves MALMO_XSD_PATH is valid)
conda activate malmo
python malmo/rl/envs/env_server.py --env bridging --port 10002 --malmo-port 10000
#   -> "Waiting for training script to connect..."

# 4. Training smoke run
conda activate train_env
python malmo/rl/training/train_sb3.py --env bridging --base-port 10002 \
    --checkpoint malmo/rl/baselines/sb3_bridging_baseline.zip
```

Or launch all three panes at once: `WorkflowScripts/launch_training.sh` (tmux).

---

## Known Issues

### `ImportError: dynamic module does not define module export function (PyInit_MalmoPython)`
The `malmo` env is not Python 3.6. The prebuilt `.so` is ABI-locked to 3.6; recreate the env
from `malmo_environment_linux.yml` (which pins `python=3.6.15`) and retry.

### `ImportError: Trying to log data to tensorboard but tensorboard is not installed`
`train_sb3.py` logs to TensorBoard. Install it into `train_env`
(`pip install tensorboard==2.20.0`); it is already in `training_environment_linux.yml`.

### SB3 checkpoint warns `Could not deserialize object clip_range / lr_schedule (FloatSchedule)`
The installed stable-baselines3 is older than the one that saved the checkpoint. Use
`stable-baselines3==2.8.0` (pinned in `training_environment_linux.yml`) — it has
`FloatSchedule` and still works with `torch==2.6.0+cu124`. Do not jump to SB3 >=2.9: it
requires `torch>=2.8`, which would replace the cu124 build and disable the GPU.

### Minecraft client crashes / "Unsupported major.minor version"
The active `java` is not Java 8. Run `sudo update-alternatives --config java` and select the
openjdk-8 entry; confirm with `java -version`.

### `MalmoPython` not found
The `.so` isn't in `malmo/Python_Examples/`. Copy it there from the release (see
[Obtaining MalmoPython](#obtaining-malmopython--the-minecraft-tree)). The RL code resolves
the binding from that folder via `BaseCFG.MALMO_PYTHON` — no copy into `site-packages` needed.
