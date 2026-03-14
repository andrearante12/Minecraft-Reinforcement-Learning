"""
simple_jump.py
--------------
Parkour Simple Jump — random agent with episode loop.

Each episode restarts the Malmo mission to respawn the agent at the start.
forceReset is disabled so Minecraft does not rebuild the world each time,
making resets much faster. The agent dies on a fall (Survival mode ends
the mission), so a full mission restart is required — teleport alone is
not sufficient after death.

Episode termination:
  - Agent Y drops below FALL_Y_THRESHOLD  →  fell (fast Y-level detection)
  - Agent Z reaches Z_SUCCESS             →  crossed the gap
  - MAX_STEPS reached                     →  timeout

Actions:
    0 - move forward
    1 - move forward + jump
    2 - strafe left
    3 - strafe right
    4 - move backward

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
import random

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MISSIONS_DIR = os.path.join(SCRIPT_DIR, 'missions')
MALMO_PYTHON = os.path.join(SCRIPT_DIR, '..', '..', 'Python_Examples')
sys.path.insert(0, os.path.abspath(MALMO_PYTHON))

try:
    import MalmoPython
except ImportError:
    print("ERROR: Could not import MalmoPython.")
    print("  Current path tried:", os.path.abspath(MALMO_PYTHON))
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_EPISODES      = 10    # total episodes to run
MAX_STEPS         = 30    # max steps per episode before timeout
STEP_DURATION     = 0.3   # seconds each action is held
SPAWN             = (0.5, 46, 0.5)  # must match XML <Placement>
FALL_Y_THRESHOLD  = 45.0  # Y below bridge surface — agent is falling
Z_SUCCESS         = 5.0   # Z position that means agent crossed the gap

# Action definitions: (name, commands_on, commands_off)
ACTIONS = [
    ("move_forward",  ["move 1"],           ["move 0"]),
    ("jump_forward",  ["move 1", "jump 1"], ["move 0", "jump 0"]),
    ("strafe_left",   ["strafe -1"],         ["strafe 0"]),
    ("strafe_right",  ["strafe 1"],          ["strafe 0"]),
    ("move_backward", ["move -1"],           ["move 0"]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def print_grid(grid_data):
    if not grid_data or len(grid_data) < 9:
        return
    row_far, row_mid, row_near = grid_data[6:9], grid_data[3:6], grid_data[0:3]
    print("\n  Floor Grid (N=forward):")
    print("  Far  | {0:^12} | {1:^12} | {2:^12} |".format(*row_far))
    print("  Mid  | {0:^12} | {1:^12} | {2:^12} |".format(*row_mid))
    print("  Near | {0:^12} | {1:^12} | {2:^12} |".format(*row_near))

def get_observation(agent_host, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ws = agent_host.getWorldState()
        if ws.number_of_observations_since_last_state > 0:
            return json.loads(ws.observations[-1].text), ws
        time.sleep(0.05)
    return {}, agent_host.getWorldState()

def start_mission(agent_host, mission_xml, max_retries=3):
    """
    Start the mission. forceReset is patched to false so Minecraft
    reuses the existing world — much faster between episodes.
    The agent still respawns at the XML <Placement> position on each start.
    """
    fast_xml = mission_xml.replace('forceReset="true"', 'forceReset="false"')

    my_mission        = MalmoPython.MissionSpec(fast_xml, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    for attempt in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if attempt == max_retries - 1:
                print("ERROR: Could not start mission: {0}".format(e))
                sys.exit(1)
            print("  Retrying ({0}/{1})...".format(attempt + 1, max_retries))
            time.sleep(2)

    print("  Waiting for mission to start...", end=' ')
    ws = agent_host.getWorldState()
    while not ws.has_mission_begun:
        print(".", end="", flush=True)
        time.sleep(0.1)
        ws = agent_host.getWorldState()
        for error in ws.errors:
            print("\n  Mission error:", error.text)
    print(" started!")
    return ws

def take_action(agent_host, action_idx):
    """Send action commands, hold for STEP_DURATION, then release."""
    _, cmds_on, cmds_off = ACTIONS[action_idx]
    for cmd in cmds_on:
        agent_host.sendCommand(cmd)
    time.sleep(STEP_DURATION)
    for cmd in cmds_off:
        agent_host.sendCommand(cmd)

def get_reward(obs):
    """
    Reward and termination based on agent position.
      -10 + done  if Y < FALL_Y_THRESHOLD  (fell off bridge)
      +10 + done  if Z >= Z_SUCCESS         (crossed the gap)
        0          otherwise
    """
    y = obs.get('YPos', SPAWN[1])
    z = obs.get('ZPos', SPAWN[2])
    if y < FALL_Y_THRESHOLD:
        return -10, True
    if z >= Z_SUCCESS:
        return +10, True
    return 0, False

# ── Init ──────────────────────────────────────────────────────────────────────
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    sys.exit(1)

xml_path = os.path.join(MISSIONS_DIR, 'parkour_simple_jump.xml')
if not os.path.exists(xml_path):
    print("ERROR: Mission file not found at:", xml_path)
    sys.exit(1)
with open(xml_path, 'r') as f:
    mission_xml = f.read()

print("=" * 50)
print("Parkour Random Agent — {0} episodes".format(NUM_EPISODES))
print("=" * 50)

# ── Episode loop ──────────────────────────────────────────────────────────────
results = []

for episode in range(NUM_EPISODES):
    print("\n--- Episode {0}/{1} ---".format(episode + 1, NUM_EPISODES))

    # Always restart the mission each episode.
    # In Survival mode the agent dies on a fall, which ends the mission.
    # startMission() respawns the agent at <Placement> from the XML.
    world_state = start_mission(agent_host, mission_xml)
    time.sleep(0.5)  # let agent fully spawn before sending commands

    obs, world_state = get_observation(agent_host)
    print("  Spawn -> X:{0:.2f} Y:{1:.2f} Z:{2:.2f}".format(
        obs.get('XPos', 0), obs.get('YPos', 0), obs.get('ZPos', 0)))

    total_reward = 0
    outcome      = "timeout"
    step         = 0

    for step in range(MAX_STEPS):
        # Mission may have ended unexpectedly (time limit, etc.)
        if not world_state.is_mission_running:
            outcome = "mission_ended"
            break

        action_idx  = random.randint(0, len(ACTIONS) - 1)
        action_name = ACTIONS[action_idx][0]

        take_action(agent_host, action_idx)

        obs, world_state = get_observation(agent_host)
        reward, done = get_reward(obs)
        total_reward += reward

        x = obs.get('XPos', 0)
        y = obs.get('YPos', 0)
        z = obs.get('ZPos', 0)
        print("  Step {0:02d} | {1:<15} | X:{2:.2f} Y:{3:.2f} Z:{4:.2f} | r={5}".format(
            step + 1, action_name, x, y, z, reward))

        if done:
            outcome = "fell" if reward < 0 else "landed"
            # Stop all movement
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("jump 0")
            agent_host.sendCommand("strafe 0")
            
            # Force-quit the mission immediately instead of waiting for death
            agent_host.sendCommand("quit")
            time.sleep(0.3)
            break

    print("  Episode {0} done — steps:{1} reward:{2} outcome:{3}".format(
        episode + 1, step + 1, total_reward, outcome))
    results.append((episode + 1, step + 1, total_reward, outcome))

    # Pause between episodes so Minecraft stabilises before next startMission
    time.sleep(1)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Run Summary")
print("=" * 50)
print("  {0:<10} {1:<8} {2:<8} {3}".format("Episode", "Steps", "Reward", "Outcome"))
for ep, steps, reward, outcome in results:
    print("  {0:<10} {1:<8} {2:<8} {3}".format(ep, steps, reward, outcome))

landed = sum(1 for _, _, _, o in results if o == "landed")
print("\n  Landed: {0}/{1} episodes".format(landed, NUM_EPISODES))