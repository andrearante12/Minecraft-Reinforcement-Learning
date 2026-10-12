"""
tests/smoke_live_split.py
-------------------------
Offline smoke test for the live split-screen viewer (no Malmo), run in train_env:

    conda run -n train_env python malmo/rl/tests/smoke_live_split.py

Checks:
  1. Playback, vector Dreamer: live_split.py --mock streams a "vector" schema and
     frames carrying cam / probs / fan (K paths) / aleatoric / epistemic.
  2. Playback, DreamerVideo: "video" schema; frames carry cam / recon / dream strip.
  3. Training hook: train.py --viz-port with EnvClient swapped for the mock env —
     frames stream with mode="train", the "update" status fires around agent.update(),
     update stats (losses, uncertainty, horizon) arrive on the stats channel, and the
     page (Inference | Training tabs) + static assets are served.
  4. Zero cost when nobody watches: with no browser connected, train.py never
     builds a frame.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import http.client
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # malmo/rl
sys.path.insert(0, ROOT)

LIVE_SPLIT = os.path.join(ROOT, "visualization", "live_split.py")


def read_events(port, want_frames, timeout=60.0, want_status=None):
    """Connect to /events; return (schema, frames, statuses)."""
    deadline = time.time() + timeout
    while True:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            conn.request("GET", "/events")
            resp = conn.getresponse()
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)
    schema, frames, statuses, event = None, [], [], None
    frames_stats = read_events.stats = []
    while time.time() < deadline:
        line = resp.fp.readline().decode("utf-8").strip()
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:])
            if event == "schema":
                schema = data
            elif event == "frame":
                frames.append(data)
            elif event == "status":
                statuses.append(data["phase"])
            elif event == "stats":
                statuses.append("stats:" + data["kind"])
                frames_stats.append(data)
        done = len(frames) >= want_frames and (want_status is None or want_status in statuses)
        if done:
            break
    conn.close()
    return schema, frames, statuses


def check_playback(env, algo, port):
    proc = subprocess.Popen([sys.executable, LIVE_SPLIT, "--env", env, "--algo", algo, "--mock",
                             "--no-browser", "--http-port", str(port), "--max-steps", "400"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        schema, frames, _ = read_events(port, want_frames=4)
    finally:
        proc.kill()
        proc.wait()
    assert schema is not None, "no schema event"
    assert len(frames) >= 4, "only {0} frames".format(len(frames))
    f = frames[-1]
    for key in ("cam", "probs", "action_name", "value", "episode", "step", "mode"):
        assert f.get(key) is not None, "frame missing {0}".format(key)
    assert f["mode"] == "eval"
    return schema, f


def check_vector_playback():
    schema, f = check_playback("hunting", "dreamer", 8791)
    assert schema["kind"] == "vector" and schema["env"] == "hunting", schema.get("kind")
    assert schema["ensemble_k"] == 5
    td = f["topdown"]
    assert len(td["members"]) == 5 and len(td["agent"]) == 2 and len(td["plan"]) == 15
    for m in td["members"]:
        assert len(m["agent"]) == 16 and len(m["sigma"]) == 15
        assert m["pig"] is None or len(m["pig"]) == 15
    assert 0.0 <= f["aleatoric"] <= 1.0 and 0.0 <= f["epistemic"] <= 1.0
    assert "activations" in f and "inputs" in f
    print("  [ok] playback vector: top-down K=5 x H=15, aleatoric/epistemic, activations")


def check_video_playback():
    schema, f = check_playback("hunting_video", "dreamer_video", 8792)
    assert schema["kind"] == "video"
    assert f["recon"] and f["real_small"]
    assert len(f["dream"]) == schema["dream_horizon"]
    assert all(d["img"] and 0.0 <= d["cont"] <= 1.0 for d in f["dream"])
    print("  [ok] playback video: recon + {0}-step dream strip".format(len(f["dream"])))


# ── training hook ─────────────────────────────────────────────────────────────
_TMP_DIRS = []


def _patch_training(cfg, step_delay):
    import training.train as train_mod
    from visualization.live_split import MockEnv

    class TrainMockEnv(MockEnv):
        def __init__(self, input_size, port=None, video=False, frame_shape=None):
            MockEnv.__init__(self, cfg, video)

        def switch_env(self, name):
            return {}

        def step(self, action):
            time.sleep(step_delay)
            return MockEnv.step(self, action)

    tmp = tempfile.mkdtemp(prefix="smoke_live_")
    _TMP_DIRS.append(tmp)
    cfg.CHECKPOINT_DIR = tmp
    cfg.LOG_DIR = tmp
    cfg.TOTAL_EPISODES = 2
    cfg.SAVE_EVERY = 1000
    cfg.LEARNING_STARTS = 60
    cfg.REAL_TRAIN_EVERY = 40
    cfg.WM_BATCH_SIZE = 32
    cfg.IMAG_BATCH = 16
    train_mod.EnvClient = TrainMockEnv
    return train_mod


def check_training_hook():
    from training.train import ENV_REGISTRY
    import visualization.live_frames as lf

    cfg = ENV_REGISTRY["hunting"][1]
    train_mod = _patch_training(cfg, step_delay=0.02)

    sys.argv = ["train.py", "--env", "hunting", "--algo", "dreamer", "--viz-port", "8793"]
    t = threading.Thread(target=train_mod.train, daemon=True)
    t.start()
    schema, frames, statuses = read_events(8793, want_frames=10, want_status="stats:update", timeout=90)
    upd = [d for d in read_events.stats if d["kind"] == "update"][0]
    for key in ("update", "env_steps", "buffer", "wm_loss", "aleatoric", "epistemic", "eff_horizon", "imag_return"):
        assert key in upd, "update stats missing {0}".format(key)

    conn = http.client.HTTPConnection("127.0.0.1", 8793, timeout=10)
    for path in ("/", "/js/live.js", "/js/net-viz.js", "/css/site.css"):
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read()
        assert r.status == 200, "{0} -> {1}".format(path, r.status)
        if path == "/":
            for el in (b'id="panel-inference"', b'id="panel-training"', b'id="charts"', b'id="topdown"'):
                assert el in body, "live.html missing {0}".format(el)
    conn.request("GET", "/../README.md")
    r = conn.getresponse()
    r.read()
    assert r.status == 404, "path traversal not blocked"
    conn.close()
    t.join(timeout=120)

    assert schema["kind"] == "vector"
    assert len(frames) >= 10 and all(f["mode"] == "train" for f in frames)
    assert "update" in statuses, statuses
    print("  [ok] train.py --viz-port: {0} train frames, statuses {1}, page served".format(
        len(frames), sorted(set(statuses))))

    # Nobody watching -> no frame builds at all.
    calls = {"n": 0}
    orig = lf.VectorDreamerFrames.build

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    lf.VectorDreamerFrames.build = counting
    cfg.TOTAL_EPISODES = 1
    _patch_training(cfg, step_delay=0.0)
    sys.argv = ["train.py", "--env", "hunting", "--algo", "dreamer", "--viz-port", "8794"]
    train_mod.train()
    lf.VectorDreamerFrames.build = orig
    assert calls["n"] == 0, "built {0} frames with no client".format(calls["n"])
    print("  [ok] no browser connected -> 0 frame builds")


def main():
    print("smoke_live_split")
    check_vector_playback()
    check_video_playback()
    try:
        check_training_hook()
    finally:
        for d in _TMP_DIRS:
            shutil.rmtree(d, ignore_errors=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
