"""Offline checks for the live top-down view (tracker, server, driver state)."""
import os, sys, json, argparse, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.live_view_server import LiveState, PlanErrorTracker, serve

# ── tracker: a perfect plan has zero error; an offset plan reports the offset at the right k
H = 4
tr = PlanErrorTracker(H)
def straight(x0):                       # agent walks +x 1 block/step, pig fixed
    ag = np.array([[x0 + i, 46, 0] for i in range(H + 1)], float)
    pg = np.array([[5, 46, 5]] * (H + 1), float)
    return ag, pg
for step in range(6):
    ag, pg = straight(step)             # plan made at `step` starting where the agent really is
    tr.add_plan(ag, pg)
    tr.update(np.array([step + 1, 46, 0.0]), np.array([5, 46, 5.0]))   # reality: exactly as planned
m = tr.mean_by_k()
assert m[0] == 0.0 and all(v == 0.0 for v in m[1:]), m
tr2 = PlanErrorTracker(H)
for step in range(6):
    ag, pg = straight(step); tr2.add_plan(ag, pg)
    tr2.update(np.array([step + 1, 46, 0.5 * 1.0]), np.array([5, 46, 5.0]))  # real agent drifts 0.5 off in z
m2 = tr2.mean_by_k()
assert abs(m2[1] - 0.5) < 1e-6 and m2[H] is not None, m2     # k=1 plan was 0.5 blocks off (z offset)
assert tr2.n[1] > 0 and tr2.n[H] > 0

# ── server: html + state round-trip
st = LiveState()
srv = serve(st, 0)                      # port 0 → ephemeral
port = srv.server_address[1]
html = urllib.request.urlopen("http://127.0.0.1:%d/" % port).read().decode()
assert "Live Imagination" in html and "/state" in html
assert urllib.request.urlopen("http://127.0.0.1:%d/state" % port).read().decode() == "null"
st.publish({"a": 1})
assert json.loads(urllib.request.urlopen("http://127.0.0.1:%d/state" % port).read()) == {"a": 1}

# ── driver publishes a well-formed state on the stand-in env
import torch; torch.set_num_threads(2)
from training.configs.hunting_cfg import HuntingCFG as C
from evaluation.live_imagination import DryRunClient, load_agent, run
class Short(C):
    MAX_STEPS = 14
st2 = LiveState(); seen = []
orig = st2.publish
st2.publish = lambda s: (seen.append(s), orig(s))
a = argparse.Namespace(episodes=2, horizon=6, particles=False, step_delay=0, max_cmds=16, max_cmds_per_sec=1e9)
out = run(DryRunClient(Short, "wandering"), load_agent(Short, None), Short, a, st2)
assert len(seen) == 28 and out["commands_sent"] == 0              # particles off by default
s = seen[10]
assert len(s["plan"]["agent"]) == 7 and len(s["plan"]["pig"]) == 7 and len(s["plan"]["members"]) == 5
assert len(s["ale"]) == 6 and len(s["epi"]) == 6 and len(s["err_by_k"]) == 7
assert len(s["real"]["trail_agent"]) == s["step"] + 1 and s["ghost"] is not None
assert seen[14]["episode"] == 2 and seen[14]["last_outcome"] == "timeout"
assert out["online_plan_error_by_k"][1] is not None
json.dumps(seen[-1])
print("ALL LIVE-VIEW CHECKS PASSED")
