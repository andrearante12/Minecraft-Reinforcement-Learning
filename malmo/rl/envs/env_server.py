"""
envs/env_server.py
------------------
Generic environment server. Loads the requested environment and exposes
it over a TCP socket so the training script can call reset() and step()
from a separate Python process / conda environment.

Usage:
    conda activate malmo
    python parkour/envs/env_server.py --env simple_jump
    python parkour/envs/env_server.py --env three_block_gap

To add a new environment:
    1. Create an XML mission file and a config class
    2. Add one entry to ENV_REGISTRY
    That's it — no other changes needed.
"""

import sys
import os
import json
import socket
import struct
import argparse

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

# ── Environment registry ──────────────────────────────────────────────────────
from envs.parkour_env import ParkourEnv

from training.configs.simple_jump_cfg      import SimpleJumpCFG
from training.configs.three_block_gap_cfg  import ThreeBlockGapCFG
from training.configs.one_block_gap_cfg    import OneBlockGapCFG

ENV_REGISTRY = {
    "one_block_gap":   OneBlockGapCFG,
    "simple_jump":     SimpleJumpCFG,
    "three_block_gap": ThreeBlockGapCFG,
}

HOST = "127.0.0.1"
PORT = 9999


def send_msg(conn, data):
    msg = json.dumps(data).encode()
    conn.sendall(struct.pack(">I", len(msg)) + msg)


def recv_msg(conn):
    raw = conn.recv(4)
    if not raw:
        return None
    length = struct.unpack(">I", raw)[0]
    data   = b""
    while len(data) < length:
        data += conn.recv(length - len(data))
    return json.loads(data.decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment to serve")
    parser.add_argument("--port", type=int, default=PORT,
                        help="TCP port to listen on (default: {0})".format(PORT))
    parser.add_argument("--malmo-port", type=int, default=None,
                        help="Minecraft/Malmo client port (default: from config, usually 10000)")
    args = parser.parse_args()

    cfg = ENV_REGISTRY[args.env]
    env = ParkourEnv(cfg, malmo_port=args.malmo_port)
    print("Env server starting — env:{0}  address:{1}:{2}".format(args.env, HOST, args.port))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, args.port))
        s.listen(1)
        print("Waiting for training script to connect...")
        conn, addr = s.accept()
        print("Connected:", addr)

        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break

                cmd = msg["cmd"]

                if cmd == "reset":
                    obs = env.reset()
                    send_msg(conn, {"obs": obs.tolist()})

                elif cmd == "step":
                    obs, reward, done, info = env.step(msg["action"])
                    send_msg(conn, {
                        "obs":    obs.tolist(),
                        "reward": reward,
                        "done":   done,
                        "info":   info,
                    })

                elif cmd == "close":
                    env.close()
                    break

    print("Server closed.")


if __name__ == "__main__":
    main()
