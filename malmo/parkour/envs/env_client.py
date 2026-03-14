"""
envs/env_client.py
------------------
Drop-in replacement for ParkourEnv that communicates over a socket.
Import this in train_simple_jump.py instead of ParkourEnv directly.

Usage:
    from envs.env_client import ParkourEnvClient
    env = ParkourEnvClient()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

import json, socket, struct
import numpy as np
from training.config import CFG

HOST = "127.0.0.1"
PORT = 9999

class ParkourEnvClient:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.observation_shape = (CFG.INPUT_SIZE,)
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

    def reset(self):
        self._send({"cmd": "reset"})
        resp = self._recv()
        return np.array(resp["obs"], dtype=np.float32)

    def step(self, action):
        self._send({"cmd": "step", "action": int(action)})
        resp = self._recv()
        return (
            np.array(resp["obs"], dtype=np.float32),
            float(resp["reward"]),
            bool(resp["done"]),
            resp["info"],
        )

    def close(self):
        self._send({"cmd": "close"})
        self.sock.close()