"""
simple_jump.py
--------------
Parkour Simple Jump — standalone Malmo agent script.
Loads parkour_simple_jump.xml, spawns the agent, and executes a timed jump.

Usage:
    python simple_jump.py

Requirements:
    - Minecraft + Malmo client must be running on port 10000
    - MALMO_XSD_PATH must be set (see README)
    - parkour_simple_jump.xml must be in the missions/ folder
"""

import sys
import os
import time
import json

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from anywhere by resolving paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MISSIONS_DIR = os.path.join(SCRIPT_DIR, 'missions')

# Add Malmo's Python_Examples to path so MalmoPython can be found
MALMO_PYTHON = os.path.join(SCRIPT_DIR, '..', '..', 'Python_Examples')
sys.path.insert(0, os.path.abspath(MALMO_PYTHON))

try:
    import MalmoPython
except ImportError:
    print("ERROR: Could not import MalmoPython.")
    print("  Make sure MALMO_PYTHON path is correct in this script.")
    print("  Current path tried:", os.path.abspath(MALMO_PYTHON))
    sys.exit(1)

# ── Helper: print the 3x3 floor grid around the agent ────────────────────────
def print_grid(grid_data):
    if not grid_data or len(grid_data) < 9:
        return
    row_far, row_mid, row_near = grid_data[6:9], grid_data[3:6], grid_data[0:3]
    print("\n--- Floor Grid (3x3, agent at center) ---")
    print("  Far  | {0:^12} | {1:^12} | {2:^12} |".format(*row_far))
    print("  Mid  | {0:^12} | {1:^12} | {2:^12} |".format(*row_mid))
    print("  Near | {0:^12} | {1:^12} | {2:^12} |".format(*row_near))
    print("-" * 50)

# ── Helper: wait for a fresh observation ─────────────────────────────────────
def get_observation(agent_host, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            return json.loads(world_state.observations[-1].text), world_state
        time.sleep(0.05)
    return {}, agent_host.getWorldState()

# ── 1. Initialize agent host ──────────────────────────────────────────────────
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR parsing agent host:', e)
    sys.exit(1)

# ── 2. Load mission XML ───────────────────────────────────────────────────────
xml_path = os.path.join(MISSIONS_DIR, 'parkour_simple_jump.xml')
if not os.path.exists(xml_path):
    print("ERROR: Mission file not found at:", xml_path)
    sys.exit(1)

with open(xml_path, 'r') as f:
    mission_xml = f.read()

my_mission = MalmoPython.MissionSpec(mission_xml, True)
my_mission_record = MalmoPython.MissionRecordSpec()
print("Mission XML loaded:", xml_path)

# ── 3. Start mission ──────────────────────────────────────────────────────────
max_retries = 3
for attempt in range(max_retries):
    try:
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if attempt == max_retries - 1:
            print("ERROR: Could not start mission after {0} attempts: {1}".format(max_retries, e))
            print("  Is the Minecraft client running? (check port 10000)")
            sys.exit(1)
        print("Retrying mission start ({0}/{1})...".format(attempt + 1, max_retries))
        time.sleep(2)

print("Waiting for mission to start...", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("\nMission error:", error.text)
print("\nMission running!")

# ── 4. Wait for first observation (agent fully spawned) ───────────────────────
time.sleep(0.5)
obs, world_state = get_observation(agent_host)
print("Spawn position -> X: {0:.2f} | Y: {1:.2f} | Z: {2:.2f}".format(
    obs.get('XPos', 0), obs.get('YPos', 0), obs.get('ZPos', 0)))
print_grid(obs.get('floor3x3', []))

# ── 5. Execute the jump ───────────────────────────────────────────────────────
print("\n--- Executing jump sequence ---")

# Move forward to build momentum
agent_host.sendCommand("move 1")
print("Moving forward...")
time.sleep(0.4)

# Jump
agent_host.sendCommand("jump 1")
print("Jumping!")
time.sleep(0.5)

# Release jump, keep moving
agent_host.sendCommand("jump 0")
time.sleep(0.5)

# Stop
agent_host.sendCommand("move 0")

# ── 6. Check outcome ──────────────────────────────────────────────────────────
time.sleep(0.5)
obs, world_state = get_observation(agent_host)

if obs:
    y = obs.get('YPos', 0)
    z = obs.get('ZPos', 0)
    print("\n--- Result ---")
    print_grid(obs.get('floor3x3', []))
    if y < 44:
        print("FELL INTO THE VOID at Z={0:.2f}".format(z))
    else:
        print("LANDED SAFELY at Z={0:.2f}!".format(z))
else:
    print("No observation received after jump.")

# ── 7. Run until mission ends ─────────────────────────────────────────────────
print("\nMonitoring until mission ends (Ctrl+C to stop early)...")
try:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obs = json.loads(world_state.observations[-1].text)
            x = obs.get('XPos', 0)
            y = obs.get('YPos', 0)
            z = obs.get('ZPos', 0)
            print("\rPosition -> X: {0:.2f} | Y: {1:.2f} | Z: {2:.2f}".format(x, y, z), end='', flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped by user.")

print("\nMission ended.")