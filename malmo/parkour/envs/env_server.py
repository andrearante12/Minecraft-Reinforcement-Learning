"""
envs/env_server.py
------------------
Runs ParkourEnv as a socket server.

Usage:
    conda activate malmo
    python parkour/envs/env_server.py
"""

import sys, os, json, socket, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from envs.parkour_env import ParkourEnv
from training.config  import CFG

HOST = "127.0.0.1"
PORT = 9999

def send_msg(conn, data):
    """Send length-prefixed JSON message."""
    msg = json.dumps(data).encode()
    conn.sendall(struct.pack(">I", len(msg)) + msg)

def recv_msg(conn):
    """Receive length-prefixed JSON message."""
    raw = conn.recv(4)
    if not raw:
        return None
    length = struct.unpack(">I", raw)[0]
    data   = b""
    while len(data) < length:
        data += conn.recv(length - len(data))
    return json.loads(data.decode())

def main():
    env = ParkourEnv(CFG)
    print("Env server starting on {0}:{1}".format(HOST, PORT))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
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