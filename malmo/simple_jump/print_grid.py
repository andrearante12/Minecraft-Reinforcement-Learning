import malmo.run_mission as run_mission
from malmo import MalmoPython
import json
import time
import sys
import os

# 1. INITIALIZE AGENT HOST
agent_host = MalmoPython.AgentHost()
try:
    # Use empty list for script run
    agent_host.parse([]) 
except RuntimeError as e:
    print('ERROR:', e)
    sys.exit(1)

# 2. LOAD YOUR XML
xml_file = 'parkour_simple_jump.xml'
if not os.path.exists(xml_file):
    print("Error: {0} not found!".format(xml_file))
    sys.exit(1)

with open(xml_file, 'r') as f:
    parkour_xml = f.read()

my_mission = MalmoPython.MissionSpec(parkour_xml, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# 3. START MISSION
try:
    agent_host.startMission(my_mission, my_mission_record)
except RuntimeError as e:
    print("Error starting mission: {0}".format(e))
    sys.exit(1)

print("Waiting for mission to start...", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
print("Running!")

# 4. GRID MONITORING FUNCTIONS
def print_grid(grid_data):
    if not grid_data or len(grid_data) < 9: return
    row3, row2, row1 = grid_data[6:9], grid_data[3:6], grid_data[0:3]
    print("\n--- Agent Radar (3x3) ---")
    print("| {0:^10} | {1:^10} | {2:^10} |".format(row3[0], row3[1], row3[2]))
    print("| {0:^10} | {1:^10} | {2:^10} |".format(row2[0], row2[1], row2[2]))
    print("| {0:^10} | {1:^10} | {2:^10} |".format(row1[0], row1[1], row1[2]))
    print("--------------------------")

# 5. MAIN LOOP
try:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get("floor3x3", [])
            if grid:
                print_grid(grid)
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nStopped.")