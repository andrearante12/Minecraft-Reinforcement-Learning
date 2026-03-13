# Malmo Windows Installation Guide

This doc contains the additional fixes I had to do to get it working on my machine. For the full original install instructions view `.\Malmo\install_windows.md`.

## Prerequisites

Install the following before running the Malmo install script. Open **PowerShell as Administrator** and run:

```powershell
winget install 7zip.7zip
winget install Gyan.FFmpeg
```

### Python 3.7 (Required)

Malmo 0.37.0 specifically targets Python 3.7. Download and install it from:
https://www.python.org/downloads/release/python-3718/. Make sure to download `Windows x86-64 executable installer` specifically as we need 64 bit to be compatible.

During installation, check **"Add Python 3.7 to PATH"**.

---

## Installation Steps

### 1. Unblock the scripts

Open **PowerShell as Administrator** and run (adjust path to match your download location):

```powershell
Unblock-File ".\Malmo\scripts\malmo_install.ps1"

Unblock-File ".\Malmo\scripts\pslib\malmo_lib.psm1"
```

### 2. Run the install script (as Administrator)

```powershell
cd ".\Malmo\scripts"

.\malmo_install.ps1
```

### 3. Verify the installation

1. Launch the Malmo client

```powershell
cd Malmo\Minecraft
.\launchClient.bat
```

2. Start a second PowerShell and start an agent:
```powershell
cd Malmo\Python_Examples
python tabular_q_learning.py
```
