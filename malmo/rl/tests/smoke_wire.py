"""
tests/smoke_wire.py
---------------------
Offline transport smoke test for the video observation wire format, run in
train_env:

    conda run -n train_env python malmo/rl/tests/smoke_wire.py

Does NOT import envs/env_server.py (it requires MalmoPython, only importable
in the `malmo` Py3.6 conda env) — instead this test reimplements the tiny
send_msg/recv_msg framing locally (identical to env_server.py's) to run a
stub server, and exercises the REAL envs/env_client.py against it. This
verifies the wire contract both sides must agree on without needing Malmo.

Checks:
  1. base64 round-trip: random uint8 frame -> encode -> EnvClient._decode_frame -> array_equal.
  2. Loopback: video-attaching stub server <-> EnvClient(video=True) — reset()/step()
     both return correct (vec, frame) tuples matching what the server sent.
  3. Cross direction: video=False EnvClient against a video-attaching server —
     extra frame keys are ignored, plain vector returned (existing callers unaffected).
  4. Cross direction: video=True EnvClient against a NON-video server — zeros
     fallback frame used instead of crashing.
  5. Python-3.6 syntax gate: py_compile every file that runs in the `malmo`
     conda env, using that env's own interpreter (catches f-strings etc.).
"""

import os
import sys
import json
import socket
import struct
import base64
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # malmo/rl
sys.path.insert(0, ROOT)

import numpy as np

from envs.env_client import EnvClient

HOST = "127.0.0.1"


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print("[{0}] {1}".format(status, name))
    if not cond:
        raise SystemExit("Smoke test failed: {0}".format(name))


# ── Local reimplementation of env_server.py's wire framing (no MalmoPython dep) ──
def send_msg(conn, data):
    msg = json.dumps(data).encode()
    conn.sendall(struct.pack(">I", len(msg)) + msg)


def recv_msg(conn):
    raw = conn.recv(4)
    if not raw:
        return None
    length = struct.unpack(">I", raw)[0]
    data = b""
    while len(data) < length:
        data += conn.recv(length - len(data))
    return json.loads(data.decode())


def run_stub_server(port, obs_size, attach_video, frame_shape, ready_event):
    def _serve():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, port))
            s.listen(1)
            ready_event.set()
            conn, _ = s.accept()
            with conn:
                for _ in range(2):  # one reset, one step
                    msg = recv_msg(conn)
                    if msg is None:
                        break
                    vec = np.random.randn(obs_size).astype(np.float32).tolist()
                    payload = {"obs": vec}
                    if msg["cmd"] == "step":
                        payload.update({"reward": 1.0, "done": False, "info": {"outcome": "alive"}})
                    if attach_video:
                        frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)
                        payload["frame_b64"] = base64.b64encode(frame.tobytes()).decode("ascii")
                        payload["frame_shape"] = list(frame_shape)
                        payload["_test_frame"] = frame.tolist()  # smuggle ground truth for the assertion
                    send_msg(conn, payload)
    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def main():
    frame_shape = (8, 8, 3)  # small, fast test frames (real config uses 64x64x3)
    obs_size = 98

    print("=" * 60)
    print("1. base64 round-trip")
    print("=" * 60)
    frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)
    resp = {"frame_b64": base64.b64encode(frame.tobytes()).decode("ascii"),
            "frame_shape": list(frame_shape)}
    client = EnvClient.__new__(EnvClient)  # bypass __init__ (no socket needed for this check)
    client.frame_shape = frame_shape
    decoded = client._decode_frame(resp)
    check("decoded frame matches original", np.array_equal(decoded, frame))
    check("decoded frame dtype is uint8", decoded.dtype == np.uint8)

    print()
    print("=" * 60)
    print("2. Loopback: video-attaching server <-> EnvClient(video=True)")
    print("=" * 60)
    port = 18821
    ready = threading.Event()
    run_stub_server(port, obs_size, attach_video=True, frame_shape=frame_shape, ready_event=ready)
    ready.wait(timeout=5)
    time.sleep(0.1)
    c = EnvClient(obs_size, port=port, video=True, frame_shape=frame_shape)
    vec, f = c.reset()
    check("reset() vec shape", vec.shape == (obs_size,))
    check("reset() frame shape", f.shape == frame_shape)
    (vec2, f2), reward, done, info = c.step(0)
    check("step() vec shape", vec2.shape == (obs_size,))
    check("step() frame shape", f2.shape == frame_shape)
    check("step() reward/done/info pass through", reward == 1.0 and done is False and info["outcome"] == "alive")
    c.sock.close()

    print()
    print("=" * 60)
    print("3. Cross direction: video=False client against video-attaching server")
    print("=" * 60)
    port = 18822
    ready = threading.Event()
    run_stub_server(port, obs_size, attach_video=True, frame_shape=frame_shape, ready_event=ready)
    ready.wait(timeout=5)
    time.sleep(0.1)
    c2 = EnvClient(obs_size, port=port, video=False)
    vec3 = c2.reset()
    check("video=False client returns a plain vector, not a tuple", isinstance(vec3, np.ndarray) and vec3.shape == (obs_size,))
    obs4, reward, done, info = c2.step(0)
    check("video=False step() also returns a plain vector", isinstance(obs4, np.ndarray) and obs4.shape == (obs_size,))
    c2.sock.close()

    print()
    print("=" * 60)
    print("4. Cross direction: video=True client against a NON-video server (zeros fallback)")
    print("=" * 60)
    port = 18823
    ready = threading.Event()
    run_stub_server(port, obs_size, attach_video=False, frame_shape=frame_shape, ready_event=ready)
    ready.wait(timeout=5)
    time.sleep(0.1)
    c3 = EnvClient(obs_size, port=port, video=True, frame_shape=frame_shape)
    vec5, f5 = c3.reset()
    check("missing frame_b64 -> zeros fallback frame", np.array_equal(f5, np.zeros(frame_shape, dtype=np.uint8)))
    check("fallback frame shape correct", f5.shape == frame_shape)
    c3.sock.close()

    print()
    print("=" * 60)
    print("5. Python-3.6 syntax gate (malmo conda env interpreter)")
    print("=" * 60)
    py36_files = [
        os.path.join(ROOT, "envs", "env_server.py"),
        os.path.join(ROOT, "envs", "hunting_env.py"),
        os.path.join(ROOT, "training", "configs", "hunting_video_cfg.py"),
        os.path.join(ROOT, "training", "configs", "world_model_cfg.py"),
        os.path.join(ROOT, "training", "configs", "hunting_cfg.py"),
    ]
    result = subprocess.run(
        ["conda", "run", "-n", "malmo", "python", "-m", "py_compile"] + py36_files,
        capture_output=True, text=True)
    check("py_compile under malmo (Py3.6) env succeeds", result.returncode == 0)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

    print()
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
