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

from training.configs.simple_jump_cfg       import SimpleJumpCFG
from training.configs.three_block_gap_cfg   import ThreeBlockGapCFG
from training.configs.one_block_gap_cfg     import OneBlockGapCFG
from training.configs.diagonal_small_cfg    import DiagonalSmallCFG
from training.configs.diagonal_medium_cfg   import DiagonalMediumCFG
from training.configs.vertical_small_cfg      import VerticalSmallCFG
from training.configs.multi_jump_course_cfg   import MultiJumpCourseCFG
from training.configs.bridging_cfg            import BridgingCFG
from training.configs.bridging_1block_cfg     import Bridging1BlockCFG
from training.configs.bridging_2block_cfg     import Bridging2BlockCFG
from training.configs.bridging_3block_cfg     import Bridging3BlockCFG
from training.configs.bridging_4block_cfg     import Bridging4BlockCFG

from envs.bridging_env import BridgingEnv

ENV_REGISTRY = {
    "one_block_gap":       (ParkourEnv, OneBlockGapCFG),
    "simple_jump":         (ParkourEnv, SimpleJumpCFG),
    "three_block_gap":     (ParkourEnv, ThreeBlockGapCFG),
    "diagonal_small":      (ParkourEnv, DiagonalSmallCFG),
    "diagonal_medium":     (ParkourEnv, DiagonalMediumCFG),
    "vertical_small":      (ParkourEnv, VerticalSmallCFG),
    "multi_jump_course":   (ParkourEnv, MultiJumpCourseCFG),
    "bridging":            (BridgingEnv, BridgingCFG),
    "bridging_1block":     (BridgingEnv, Bridging1BlockCFG),
    "bridging_2block":     (BridgingEnv, Bridging2BlockCFG),
    "bridging_3block":     (BridgingEnv, Bridging3BlockCFG),
    "bridging_4block":     (BridgingEnv, Bridging4BlockCFG),
    "bridging_5block":     (BridgingEnv, BridgingCFG),
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

    EnvClass, cfg = ENV_REGISTRY[args.env]
    env = EnvClass(cfg, malmo_port=args.malmo_port)
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
                    try:
                        if msg.get("force_reset") and hasattr(env, '_next_force_reset'):
                            env._next_force_reset = True
                        obs = env.reset()
                        send_msg(conn, {"obs": obs.tolist()})
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print("ERROR during reset: {0}".format(e))
                        send_msg(conn, {"error": str(e)})

                elif cmd == "step":
                    try:
                        obs, reward, done, info = env.step(msg["action"])
                        send_msg(conn, {
                            "obs":    obs.tolist(),
                            "reward": reward,
                            "done":   done,
                            "info":   info,
                        })
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print("ERROR during step: {0}".format(e))
                        send_msg(conn, {"error": str(e)})

                elif cmd == "switch_env":
                    env_name = msg["env"]
                    if env_name not in ENV_REGISTRY:
                        send_msg(conn, {"error": "Unknown env: {0}".format(env_name)})
                    else:
                        try:
                            env.close()
                        except Exception:
                            pass
                        EnvClass, cfg = ENV_REGISTRY[env_name]
                        env = EnvClass(cfg, malmo_port=args.malmo_port, force_reset=True)
                        send_msg(conn, {"status": "ok", "env": env_name})

                elif cmd == "close":
                    env.close()
                    break

    print("Server closed.")


if __name__ == "__main__":
    main()
