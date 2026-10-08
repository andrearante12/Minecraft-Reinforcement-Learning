"""Offline checks for the in-game particle overlay (no Malmo needed)."""
import os, re, sys
import numpy as np
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.configs.hunting_cfg import HuntingCFG as C
from visualization.ingame_overlay import build_commands, probe_commands, Throttle, STYLE
from evaluation.imagination_vs_reality import decode_positions

PAT = re.compile(r"^/particle (\w+) (-?\d+\.\d\d) (-?\d+\.\d\d) (-?\d+\.\d\d) 0 0 0 0 \d+ force$")

# synthetic imagined sequence: agent walks +z, pig stands 4 blocks ahead
T = 13
seq = np.zeros((T, C.INPUT_SIZE), np.float32)
p = C.PROPRIOCEPTION_SIZE
for t in range(T):
    seq[t, 8] = 0.5 * t                  # agent z rel spawn
    seq[t, p + 2] = 4.0 - 0.5 * t        # pig dz → pig fixed at world z = spawn.z + 4
members = np.stack([seq + np.random.RandomState(k).normal(0, .1, seq.shape) * (np.arange(C.INPUT_SIZE) == p)
                    for k in range(5)])

cmds = build_commands(seq, C, members_obs=members, max_cmds=40)
assert all(PAT.match(c) for c in cmds), [c for c in cmds if not PAT.match(c)]
names = [PAT.match(c).group(1) for c in cmds]
assert set(names) == set(STYLE.values()), set(names)

# coordinates round-trip: every flame particle sits at the imagined pig (x,z), body height
ag, pg = decode_positions(seq, C)
flames = [PAT.match(c).groups() for c in cmds if c.startswith("/particle flame")]
assert flames and all(abs(float(f[1]) - pg[0][0]) < .01 and abs(float(f[3]) - pg[0][2]) < .01 for f in flames)
assert all(abs(float(f[2]) - (pg[0][1] + 0.6)) < .01 for f in flames)
# agent trail advances along +z
trail = [float(PAT.match(c).group(4)) for c in cmds if "happyVillager" in c]
assert trail == sorted(trail) and trail[-1] > trail[0]

# budget + priority: with a tiny budget the ghost box wins over path/fan/agent
few = build_commands(seq, C, members_obs=members, max_cmds=5)
assert len(few) == 5 and all("endRod" in c for c in few)
assert len(build_commands(seq, C, members_obs=members, max_cmds=16)) <= 16

# probe: one marker per style name, in front of spawn
pc = probe_commands(C.SPAWN)
assert len(pc) == len(STYLE) and all(PAT.match(c) for c in pc)

# throttle: burst limited to rate, refills with time
t = [0.0]
th = Throttle(10, clock=lambda: t[0])
assert th.take(25) == 10 and th.take(5) == 0
t[0] = 0.5; assert th.take(100) == 5

# mission XML injection (same replace as HuntingEnv.reset) stays valid, handler under AgentHandlers
xml = open(C.MISSION_FILE).read().replace('</AgentHandlers>', '  <ChatCommands/>\n    </AgentHandlers>')
ns = {"m": "http://ProjectMalmo.microsoft.com"}
root = ET.fromstring(xml)
assert root.find(".//m:AgentHandlers/m:ChatCommands", ns) is not None

# end-to-end dry run through the live driver on the stand-in env
from evaluation.live_imagination import DryRunClient, load_agent, run
import argparse
a = argparse.Namespace(episodes=1, horizon=8, particles=True, step_delay=0, max_cmds=16, max_cmds_per_sec=1e9)
import torch; torch.set_num_threads(2)
class Short(C):
    MAX_STEPS = 24
client = DryRunClient(Short, "penned")
out = run(client, load_agent(Short, None), Short, a)
assert out["commands_sent"] > 0 and len(client.sent) > 0 and all(PAT.match(c) for c in client.sent[0])
assert max(len(s) for s in client.sent) <= 16
assert out["windows"] >= 0
print("ALL IN-GAME OVERLAY CHECKS PASSED")
