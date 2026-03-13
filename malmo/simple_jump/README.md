# Windows Installation Guide

# Malmo Windows Installation Guide
**Version:** Malmo 0.37.0 (Windows 64-bit with Boost, Python 3.7)

---

## Prerequisites

Install the following before running the Malmo install script. Open **PowerShell as Administrator** and run:

```powershell
winget install 7zip.7zip
winget install Gyan.FFmpeg
```

> **Note:** If `winget` is unavailable, install manually:
> - **7-Zip:** https://www.7-zip.org/download.html
> - **FFmpeg:** https://ffmpeg.org/download.html — add `ffmpeg/bin` to your system `PATH`

### Python 3.7 (Required)

Malmo 0.37.0 specifically targets Python 3.7. Download and install it from:
https://www.python.org/downloads/release/python-3718/

During installation, check **"Add Python 3.7 to PATH"**.

---

## Installation Steps

### 1. Unblock the scripts

Open **PowerShell as Administrator** and run (adjust path to match your download location):

```powershell
Unblock-File "C:\Users\<your-username>\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\scripts\malmo_install.ps1"

Unblock-File "C:\Users\<your-username>\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\scripts\pslib\malmo_lib.psm1"
```

> **Why this is needed:** Windows blocks scripts downloaded from the internet by default. The install script's built-in self-elevation to admin fails silently without this step.

### 2. Run the install script (as Administrator)

```powershell
cd "C:\Users\<your-username>\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\scripts"

.\malmo_install.ps1
```

> **Critical:** PowerShell must be launched as Administrator. Right-click the PowerShell icon → **"Run as Administrator"**. Running the script from a non-elevated shell causes the silent dependency installation failure seen below.

---

### 3. Verify the installation

Note: Make sure to replace filepath with the correct filepath to your local install

1. Launch the Malmo client
```
cd Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Minecraft
.\launchClient.bat
```
2. Start a second PowerShell and start an agent:
```
cd Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples
python tabular_q_learning.py
```
