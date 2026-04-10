# launch_training.ps1
# Opens 3 terminal tabs in one Admin window, each pre-configured for its role.

$projectRoot = Split-Path -Parent $PSScriptRoot

# --- Command Definitions ---

# Tab 1: Minecraft client
$mc_cmd = "Set-Location '$projectRoot'; Set-Location '.\Malmo\Minecraft'; .\launchClient.bat"
$mc_enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($mc_cmd))

# Tab 2: Environment server (Conda malmo)
# Note: Ensure 'conda init powershell' has been run once on your system
$env_cmd = "Set-Location '$projectRoot'; conda activate malmo"
$env_enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($env_cmd))

# Tab 3: Training (Conda train_env)
$train_cmd = "Set-Location '$projectRoot'; conda activate train_env"
$train_enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($train_cmd))

# --- Launch Execution ---

# We chain the commands using ';' so Windows Terminal opens them as tabs in one window.
# -Verb RunAs is the secret sauce that forces the Administrator prompt.

Start-Process wt -ArgumentList (
    "new-tab --title `"MC-Client`" powershell.exe -NoExit -EncodedCommand $mc_enc",
    "; new-tab --title `"EnvServer`" powershell.exe -NoExit -EncodedCommand $env_enc",
    "; new-tab --title `"Training`" powershell.exe -NoExit -EncodedCommand $train_enc"
) -Verb RunAs