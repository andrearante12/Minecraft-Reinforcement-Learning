# Setup & Installation

## Table of Contents

- [Conda Environment Setup](#conda-environment-setup)
- [Malmo Installation (Windows)](#malmo-installation-windows)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Verify the Installation](#verify-the-installation)
- [Known Issues](#known-issues)

---

## Conda Environment Setup

This repo requires two separate conda environments. This is because Malmo requires an older version of Python, while the training scripts utilize a newer version of Python in order to support PyTorch and other dependencies. A web socket is used to transfer data between the two environments.

Both environment definitions are stored in `.\conda_environments\`.

### 1. **malmo** — runs Minecraft and the environment server

```powershell
conda env create -f .\conda_environments\malmo_environment.yml
conda activate malmo
```

After creating the malmo environment, copy `MalmoPython.pyd` into the environment's site-packages so it can be imported:

```powershell
copy ".\Malmo\Python_Examples\MalmoPython.pyd" "$env:CONDA_PREFIX\Lib\site-packages\"
```

### 2. **train_env** — runs RL training

In a separate terminal tab:

```powershell
conda env create -f .\conda_environments\training_environment.yml
conda activate train_env
```

### Environment Variables

Set these once — they persist across all future terminal sessions. Open PowerShell as Administrator and run:

```powershell
# Malmo XSD schema path (required for mission validation)
[System.Environment]::SetEnvironmentVariable("MALMO_XSD_PATH",
    "C:\Users\<user>\Desktop\Minecraft-Reinforcement-Learning\Malmo\Schemas", "Machine")

# Makes the parkour package importable from any directory
[System.Environment]::SetEnvironmentVariable("PYTHONPATH",
    "C:\Users\<user>\Desktop\Minecraft-Reinforcement-Learning\Malmo\\rl", "User")
```

Replace `<user>` with your Windows username and ensure the full file path is accurate. Open a fresh PowerShell after setting these and verify:

```powershell
echo $env:MALMO_XSD_PATH
echo $env:PYTHONPATH
```

---

## Malmo Installation (Windows)

This doc contains the additional fixes required to get Malmo working on Windows. For the full original install instructions view [Official Install (Windows)](../../offical_install/install_windows.md).

### Prerequisites

Install the following before running the Malmo install script. Open **PowerShell as Administrator** and run:

```powershell
winget install 7zip.7zip
winget install Gyan.FFmpeg
```

### Python 3.7 (Required)

Malmo 0.37.0 specifically targets Python 3.7. Download and install it from:
https://www.python.org/downloads/release/python-3718/

Make sure to download the `Windows x86-64 executable installer` — the 64-bit version is required to be compatible with Malmo's DLLs.

During installation, check **"Add Python 3.7 to PATH"**.

---

## Installation Steps

### 1. Unblock the scripts

Open **PowerShell as Administrator** and run:

```powershell
Unblock-File ".\Malmo\scripts\malmo_install.ps1"
Unblock-File ".\Malmo\scripts\pslib\malmo_lib.psm1"
```

### 2. Run the install script (as Administrator)

```powershell
cd ".\Malmo\scripts"
.\malmo_install.ps1
```

### Verify the Installation

1. Launch the Malmo client:

```powershell
cd Malmo\Minecraft
.\launchClient.bat
```

2. In a separate terminal, run an example agent:

```powershell
cd Malmo\Python_Examples
python tabular_q_learning.py
```

---

## Known Issues

### `Error while installing dependencies` during `malmo_install.ps1`

The install script tries to self-elevate to Administrator and fails silently. Fix: launch PowerShell as Administrator before running the script, and unblock both `.ps1` and `.psm1` files first (see [Installation Steps](#installation-steps)).

### `MalmoPython` not found in malmo conda environment

`MalmoPython.pyd` is not installed as a package — it must be manually copied into the environment's `site-packages`. Fix: see [Conda Environment Setup](#conda-environment-setup).
