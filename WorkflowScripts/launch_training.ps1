# launch_training.ps1
# Opens 3 terminal windows from the project root, each pre-configured for its role.
# Run this script from anywhere — it resolves the project root automatically.

$projectRoot = Split-Path -Parent $PSScriptRoot

# Terminal 1: Minecraft client
# Encode the command to prevent wt from splitting on semicolons
$mc_cmd = "Set-Location '$projectRoot'; Set-Location '.\Malmo\Minecraft'; .\launchClient.bat"
$mc_encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($mc_cmd))
Start-Process wt -ArgumentList @(
    "new-tab",
    "--title", "MC-Client",
    "--",
    "powershell.exe", "-NoExit", "-EncodedCommand", $mc_encoded
)

Start-Sleep -Milliseconds 500

# Terminal 2: Environment server (malmo env, Python 3.7)
# Uses cmd.exe so conda activate works without needing the PowerShell shell hook
Start-Process wt -ArgumentList @(
    "new-tab",
    "--title", "EnvServer",
    "--",
    "cmd.exe", "/K", "cd /d `"$projectRoot`" && conda activate malmo"
)

Start-Sleep -Milliseconds 500

# Terminal 3: Training (train_env, Python 3.10)
Start-Process wt -ArgumentList @(
    "new-tab",
    "--title", "Training",
    "--",
    "cmd.exe", "/K", "cd /d `"$projectRoot`" && conda activate train_env"
)
