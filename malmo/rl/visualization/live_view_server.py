"""
visualization/live_view_server.py
---------------------------------
Tiny stdlib web server for the LIVE top-down imagination view. The driver
(evaluation/live_imagination.py) publishes one JSON state per real step; the
page (live_view.html) polls /state and redraws. Open http://localhost:8770 next
to the Minecraft window.

Also holds PlanErrorTracker: the running "how wrong was the plan I made k steps
ago?" statistic shown in the page.
"""

import os
import json
import threading
import numpy as np
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class LiveState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None

    def publish(self, state):
        blob = json.dumps(state)
        with self._lock:
            self._state = blob

    def get(self):
        with self._lock:
            return self._state


def serve(state, port, html_path=None):
    page = open(html_path or os.path.join(HERE, "live_view.html"), "rb").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.split("?")[0] == "/state":
                body = (state.get() or "null").encode()
                ctype = "application/json"
            else:
                body, ctype = page, "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class PlanErrorTracker:
    """Compare every past plan with reality as it unfolds.

    add_plan(agent_xyz (H+1,3), pig_xyz (H+1,3)) after planning at step t.
    update(real_agent_xyz, real_pig_xyz) after the real step: for each stored plan
    made k steps ago, error_k = max(|agent err|, |pig err|) at plan index k.
    """

    def __init__(self, horizon):
        self.H = horizon
        self.plans = deque(maxlen=horizon)          # newest last
        self.sum = np.zeros(horizon + 1)
        self.n = np.zeros(horizon + 1)

    def reset_episode(self):
        self.plans.clear()

    def add_plan(self, agent, pig):
        self.plans.append((np.asarray(agent), None if pig is None else np.asarray(pig)))

    def update(self, real_agent, real_pig):
        ghost = None
        plans = list(self.plans)
        for age, (pa, pp) in enumerate(reversed(plans), start=1):    # age = steps since planned
            if age > self.H:
                break
            e = float(np.linalg.norm(pa[age] - real_agent))
            if pp is not None and real_pig is not None:
                e = max(e, float(np.linalg.norm(pp[age] - real_pig)))
            self.sum[age] += e
            self.n[age] += 1
            if age == self.H or (ghost is None and age == len(plans)):
                ghost = {"age": age, "agent": pa[age].tolist(), "pig": None if pp is None else pp[age].tolist()}
        return ghost

    def mean_by_k(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(self.n > 0, self.sum / np.maximum(self.n, 1), np.nan)
        m[0] = 0.0
        return [None if np.isnan(v) else round(float(v), 3) for v in m]
