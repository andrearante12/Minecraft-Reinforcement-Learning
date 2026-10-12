"""
utils/scout_seeds.py
---------------------
Find a world seed whose area around the origin suits the hunting cage: land
(little water), some trees for occlusion, gentle terrain. Run in the malmo
conda env (Python 3.6) with a Minecraft client up and NO env server attached:

    conda run -n malmo python malmo/rl/utils/scout_seeds.py --seeds 4837 1 2 3 ...
    conda run -n malmo python malmo/rl/utils/scout_seeds.py --random 15

Per seed it loads the world with a creative-mode agent high above the origin
and reads one ObservationFromGrid block (a (2H+1) x 41 x (2H+1) column region,
absolute y 60..100) before the agent can fall far. For every column it takes the
top non-air block, then reports:
  water%   share of columns topped by water/ice
  tree%    share topped by leaves/logs (occlusion)
  surface  median top-block y (-> SPAWN y = surface + 1)
  rough    std of surface height over land columns (blocks)
  spawn    top block + surface y at the agent spawn column
Lower water and rough, moderate tree% (10-35%) is best.

--save-heightmap writes envs/hunting_vision/heightmaps/<seed>.json (top block
y + type per column). HuntingEnv uses it (cfg.HEIGHTMAP) to place the agent
and prey exactly on the ground instead of a fixed y.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_ROOT)
from training.configs.base_cfg import BaseCFG   # noqa: E402
sys.path.insert(0, os.path.abspath(BaseCFG.MALMO_PYTHON))
import MalmoPython   # noqa: E402

Y_LO, Y_HI = 60, 100
AGENT_Y = 130
WATER = ("water", "flowing_water", "ice")
TREE = ("leaves", "leaves2", "log", "log2")

MISSION = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About><Summary>seed scout</Summary></About>
  <ServerSection>
    <ServerInitialConditions>
      <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
      <AllowSpawning>false</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
      <DefaultWorldGenerator seed="{seed}" forceReset="true"/>
      <ServerQuitFromTimeUp timeLimitMs="20000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Creative">
    <Name>Scout</Name>
    <AgentStart><Placement x="0.5" y="{ay}" z="0.5" yaw="0" pitch="90"/></AgentStart>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ObservationFromGrid>
        <Grid name="scan">
          <min x="-{h}" y="{ylo}" z="-{h}"/>
          <max x="{h}"  y="{yhi}" z="{h}"/>
        </Grid>
      </ObservationFromGrid>
      <MissionQuitCommands/>
    </AgentHandlers>
  </AgentSection>
</Mission>'''


def scout(host, seed, half, port):
    xml = MISSION.format(seed=seed, ay=AGENT_Y, h=half,
                         ylo=Y_LO - AGENT_Y, yhi=Y_HI - AGENT_Y)
    mission = MalmoPython.MissionSpec(xml, True)
    pool = MalmoPython.ClientPool()
    pool.add(MalmoPython.ClientInfo("127.0.0.1", port))
    host.startMission(mission, pool, MalmoPython.MissionRecordSpec(), 0, "scout")
    ws = host.getWorldState()
    t0 = time.time()
    while not ws.has_mission_begun and time.time() - t0 < 90:
        time.sleep(0.2)
        ws = host.getWorldState()
    grid = None
    t0 = time.time()
    while grid is None and time.time() - t0 < 20 and ws.is_mission_running:
        ws = host.getWorldState()
        if ws.number_of_observations_since_last_state > 0:
            grid = json.loads(ws.observations[-1].text).get("scan")
        time.sleep(0.1)
    try:
        host.sendCommand("quit")
    except Exception:
        pass
    while host.getWorldState().is_mission_running:
        time.sleep(0.2)
    if grid is None:
        return None

    side, ny = 2 * half + 1, Y_HI - Y_LO + 1
    # Malmo grid order: y (outer), z, x (inner)
    blocks = np.array(grid, dtype=object).reshape(ny, side, side)
    top_y = np.full((side, side), -1)
    top_b = np.empty((side, side), dtype=object)
    for zi in range(side):
        for xi in range(side):
            col = blocks[:, zi, xi]
            solid = [i for i in range(ny) if col[i] != "air"]
            if solid:
                i = solid[-1]
                top_y[zi, xi], top_b[zi, xi] = Y_LO + i, col[i]
            else:
                top_b[zi, xi] = "air"
    flat_b = top_b.ravel()
    water = np.mean([b in WATER for b in flat_b])
    tree = np.mean([b in TREE for b in flat_b])
    land = np.array([b not in WATER and b not in TREE and b != "air" for b in flat_b])
    ys = top_y.ravel()[land]
    c = half
    hm = {"seed": seed, "half": half, "top_y": top_y.tolist(),
          "top_block": [[str(b) for b in row] for row in top_b]}
    return {"heightmap": hm,
        "seed": seed, "water": water, "tree": tree,
        "surface": float(np.median(ys)) if len(ys) else float("nan"),
        "rough": float(np.std(ys)) if len(ys) else float("nan"),
        "spawn_block": top_b[c, c], "spawn_y": int(top_y[c, c]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--random", type=int, default=0, help="also try this many random seeds")
    ap.add_argument("--half", type=int, default=21, help="cage half-width to scan (default 21)")
    ap.add_argument("--port", type=int, default=10000)
    ap.add_argument("--save-heightmap", action="store_true")
    args = ap.parse_args()
    seeds = list(args.seeds) + [str(random.randint(1, 10 ** 9)) for _ in range(args.random)]
    host = MalmoPython.AgentHost()
    print("%-11s %6s %6s %8s %6s  %s" % ("seed", "water", "tree", "surface", "rough", "spawn column"))
    for seed in seeds:
        try:
            r = scout(host, seed, args.half, args.port)
        except Exception as e:
            print("%-11s failed: %s" % (seed, e))
            time.sleep(2)
            continue
        if r is None:
            print("%-11s no observation" % seed)
            continue
        if args.save_heightmap:
            out_dir = os.path.join(RL_ROOT, "envs", "hunting_vision", "heightmaps")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            with open(os.path.join(out_dir, "%s.json" % seed), "w") as f:
                json.dump(r["heightmap"], f)
        print("%-11s %5.0f%% %5.0f%% %8.1f %6.2f  %s @ y=%d" % (
            r["seed"], 100 * r["water"], 100 * r["tree"], r["surface"], r["rough"],
            r["spawn_block"], r["spawn_y"]), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
