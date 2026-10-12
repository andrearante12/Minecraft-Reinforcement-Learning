"""
visualization/live_viz.py
-------------------------
Live neural-network visualizer. Runs a trained PPO actor-critic agent through a
rollout (against the real Malmo env server, or a synthetic ``--mock`` stream)
and broadcasts the network's internal activations to a browser dashboard over a
WebSocket.

The forward pass we visualize is the one PPO already runs inside
``select_action`` — forward hooks capture those activations as a side effect, so
there is no extra compute and the displayed neurons are exactly the ones that
chose the action.

Run (3-terminal, like evaluate.py):
    T1: cd malmo/Minecraft && ./launchClient.sh
    T2: conda activate malmo     && python malmo/rl/envs/env_server.py --env simple_jump --port 10002
    T3: conda activate train_env && python malmo/rl/visualization/live_viz.py \
            --env simple_jump --checkpoint malmo/rl/baselines/ppo_parkour_baseline.pt --port 10002
    # then open the URL it prints, e.g. http://localhost:8765/

No Malmo? Use --mock to drive the dashboard with synthetic observations.

Dump a replay JSON for the overview site:
    python malmo/rl/visualization/live_viz.py --env simple_jump --checkpoint <pt> --mock \\
        --record site/data/parkour_replay.json --record-steps 80 --no-browser
"""

import os
import sys
import json
import time
import asyncio
import argparse
import webbrowser

import numpy as np

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> malmo/rl
sys.path.insert(0, PARKOUR_ROOT)

from aiohttp import web

from models.actor_critic import ActorCritic
from algos.ppo import PPO
from envs.env_client import EnvClient
from evaluation.evaluate import ENV_REGISTRY  # reuse the parkour env registry

from visualization.activation_probe import build_actor_critic_probe
from visualization.net_schema import build_schema, FrameBuilder

WEBVIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webviz")
MOCK_STEP_DT = 0.15  # emulate Malmo's ~6.6 Hz step rate in --mock


# ── argument parsing ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Live neural-network visualizer")
    p.add_argument("--env", type=str, default="simple_jump",
                   choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained .pt checkpoint")
    p.add_argument("--port", type=int, default=9999,
                   help="TCP port of the Malmo env server (ignored with --mock)")
    p.add_argument("--http-port", type=int, default=8765,
                   help="Port to serve the dashboard on (default 8765)")
    p.add_argument("--mock", action="store_true",
                   help="Drive with synthetic obs; no Malmo / env server needed")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not auto-open the dashboard in a browser")
    p.add_argument("--record", type=str, default=None,
                   help="Write schema+frames JSON to this path, then exit")
    p.add_argument("--record-steps", type=int, default=80,
                   help="Frames to capture with --record (default 80)")
    return p.parse_args()


def _write_record(path, schema, frames):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump({"schema": schema, "frames": frames}, f)
    print("Wrote {0} frames to {1}".format(len(frames), path))


# ── frame broadcasting ─────────────────────────────────────────────────────
def _broadcast(app, frame):
    """Runs on the event-loop thread (via call_soon_threadsafe). Remember the
    latest frame and fan it out to all connected browsers.
    """
    app["latest_frame"] = frame
    for ws in list(app["clients"]):
        if not ws.closed:
            asyncio.ensure_future(ws.send_json(frame))


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    app = request.app
    app["clients"].add(ws)
    # A new client gets the topology, then the most recent frame immediately.
    await ws.send_json(app["schema"])
    if app["latest_frame"] is not None:
        await ws.send_json(app["latest_frame"])
    try:
        async for _ in ws:  # we don't expect inbound messages; just hold open
            pass
    finally:
        app["clients"].discard(ws)
    return ws


async def index_handler(request):
    return web.FileResponse(os.path.join(WEBVIZ_DIR, "index.html"))


# ── the rollout (runs in a worker thread) ──────────────────────────────────
def _normalized_input(agent, obs, cfg):
    """The obs the model actually saw — replicate PPO's one-line normalization."""
    if getattr(agent, "obs_rms", None) is not None:
        return agent.obs_rms.normalize(obs, cfg.NORM_CLIP)
    return obs


def _mock_step(obs):
    """A bounded random walk so neurons visibly move in --mock mode."""
    obs = obs + np.random.randn(*obs.shape).astype(np.float32) * 0.3
    obs = np.clip(obs, -3.0, 3.0)
    obs[0] = float(np.random.rand() < 0.85)  # on_ground mostly true
    return obs


def run_rollout(app, loop, cfg, agent, probe, fb, args):
    """Blocking rollout loop. Pushes one frame per env step to the loop."""
    env = None
    try:
        if not args.mock:
            env = EnvClient(cfg.INPUT_SIZE, port=args.port)

        while not app["stop"]:
            if args.mock:
                obs = np.zeros(cfg.INPUT_SIZE, dtype=np.float32)
            else:
                obs = env.reset()
            done = False

            while not done and not app["stop"]:
                action = agent.select_action(obs, greedy=True)  # forward → hooks fire
                snap = probe.snapshot()
                norm_in = _normalized_input(agent, obs, cfg)
                frame = fb.build(snap, action, norm_in)
                loop.call_soon_threadsafe(_broadcast, app, frame)
                if args.record:
                    app["recorded"].append(frame)
                    if len(app["recorded"]) >= args.record_steps:
                        app["stop"] = True
                        break

                if args.mock:
                    obs = _mock_step(obs)
                    time.sleep(MOCK_STEP_DT)
                    if frame["step"] % 200 == 0:
                        done = True  # roll a fresh "episode" periodically
                else:
                    obs, _reward, done, _info = env.step(action)
    except Exception as e:  # surface rollout errors without killing the server
        print("Rollout stopped:", repr(e))
    finally:
        if args.record:
            _write_record(args.record, app["schema"], app["recorded"])
        if env is not None:
            env.close()
        if args.record:
            loop.call_soon_threadsafe(lambda: sys.exit(0))


# ── server lifecycle ───────────────────────────────────────────────────────
def make_app(cfg, agent, probe, fb, args):
    app = web.Application()
    app["clients"] = set()
    app["schema"] = build_schema(cfg, agent.model)
    app["latest_frame"] = None
    app["stop"] = False
    app["recorded"] = []

    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", WEBVIZ_DIR)

    async def on_startup(app):
        loop = asyncio.get_event_loop()
        url = "http://localhost:{0}/".format(args.http_port)
        print("\nDashboard:", url, "\n")
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        # run the blocking rollout in the default thread-pool executor
        app["rollout"] = loop.run_in_executor(
            None, run_rollout, app, loop, cfg, agent, probe, fb, args)

    async def on_cleanup(app):
        app["stop"] = True
        for ws in list(app["clients"]):
            await ws.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    args = parse_args()
    cfg = ENV_REGISTRY[args.env]

    print("=" * 60)
    print("Live Neural-Network Visualizer")
    print("=" * 60)
    if args.record:
        args.no_browser = True

    print("Environment:", args.env)
    print("Checkpoint: ", args.checkpoint)
    print("Mode:       ", "MOCK (synthetic obs)" if args.mock else "LIVE (Malmo)")
    if args.record:
        print("Record:     ", args.record, "({0} steps)".format(args.record_steps))

    model = ActorCritic(cfg)
    agent = PPO(model, cfg)
    agent.load(args.checkpoint)
    model.eval()

    with build_actor_critic_probe(model) as probe:
        fb = FrameBuilder(cfg, model)
        app = make_app(cfg, agent, probe, fb, args)
        try:
            web.run_app(app, host="127.0.0.1", port=args.http_port, print=None)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
