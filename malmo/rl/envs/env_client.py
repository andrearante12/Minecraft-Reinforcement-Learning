"""
envs/env_client.py
------------------
Generic environment client. Connects to env_server.py over a TCP socket
and exposes the same reset() / step() / close() interface as ParkourEnv.

No environment-specific knowledge lives here — the client just forwards
commands and receives observations as JSON. INPUT_SIZE is passed in from
the env's config so the training script knows the observation shape.

Usage:
    from envs.env_client import EnvClient
    from training.configs.simple_jump_cfg import SimpleJumpCFG as CFG

    env = EnvClient(CFG.INPUT_SIZE)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

import json
import socket
import struct
import numpy as np

HOST = "127.0.0.1"
PORT = 9999


class EnvClient:
    def __init__(self, input_size, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.observation_shape = (input_size,)
        print("Connected to env server at {0}:{1}".format(host, port))

    def _send(self, data):
        msg = json.dumps(data).encode()
        self.sock.sendall(struct.pack(">I", len(msg)) + msg)

    def _recv(self):
        try:
            raw = self.sock.recv(4)
            if not raw:
                raise ConnectionError("Server closed the connection unexpectedly.")
            length = struct.unpack(">I", raw)[0]
            data   = b""
            while len(data) < length:
                chunk = self.sock.recv(length - len(data))
                if not chunk:
                    raise ConnectionError("Server disconnected mid-message.")
                data += chunk
            return json.loads(data.decode())
        except Exception as e:
            raise ConnectionError("recv failed: {0}".format(e))

    def reset(self, max_retries=3, force_reset=False):
        for attempt in range(max_retries):
            msg = {"cmd": "reset"}
            if force_reset:
                msg["force_reset"] = True
            self._send(msg)
            resp = self._recv()
            if "error" in resp:
                print("WARNING: reset failed (attempt {0}/{1}): {2}".format(
                    attempt + 1, max_retries, resp["error"]))
                if attempt == max_retries - 1:
                    raise RuntimeError("reset failed after {0} retries: {1}".format(
                        max_retries, resp["error"]))
                import time
                time.sleep(5)
                continue
            return np.array(resp["obs"], dtype=np.float32)

    def step(self, action):
        self._send({"cmd": "step", "action": int(action)})
        resp = self._recv()
        if "error" in resp:
            # Treat mission failure mid-episode as a terminal step
            print("WARNING: step failed (mission error), ending episode")
            dummy_obs = np.zeros(self.observation_shape, dtype=np.float32)
            return dummy_obs, -5.0, True, {"outcome": "mission_error", "steps": 0, "pos": (0, 0, 0), "action": "none"}
        return (
            np.array(resp["obs"], dtype=np.float32),
            float(resp["reward"]),
            bool(resp["done"]),
            resp["info"],
        )

    def switch_env(self, env_name):
        self._send({"cmd": "switch_env", "env": env_name})
        resp = self._recv()
        if "error" in resp:
            raise RuntimeError("switch_env failed: {0}".format(resp["error"]))
        return resp

    def close(self):
        self._send({"cmd": "close"})
        self.sock.close()
