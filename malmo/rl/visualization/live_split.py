"""
visualization/live_split.py
---------------------------
Live split-screen for a trained world-model agent: the Minecraft camera on the
left, the model's predictions on the right (site/live.html), in one browser tab.

Auto-detects the checkpoint family:
  world_model_state        → algos/dreamer.py      (policy, V, ensemble dream fan,
                                                     aleatoric / epistemic)
  video_world_model_state  → algos/dreamer_video.py (posterior decode + imagined
                                                     frame strip)

Run (Minecraft can be visible or headless under Xvfb):
    T1: cd malmo/Minecraft && xvfb-run -a -s "-screen 0 1280x720x24" ./launchClient.sh
    T2: conda activate malmo && MALMO_RECORD_VIDEO=1 \\
            python malmo/rl/envs/env_server.py --env hunting --port 9999
        # hunting_video ships its own 64x64 frames — do NOT set MALMO_RECORD_VIDEO there
    T3: conda activate train_env && python malmo/rl/visualization/live_split.py \\
            --env hunting --checkpoint malmo/rl/checkpoints/dreamer_hunting_dreamer_ep25.pt
    # then open http://localhost:8765/

No Malmo? --mock drives the page with synthetic obs + a synthetic camera
(--checkpoint optional with --mock; pass --algo to pick a random-init agent).
To watch while training instead, use training/train.py --viz-port 8765.
"""

import os
import sys
import time
import argparse
import webbrowser

import numpy as np
import torch

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> malmo/rl
sys.path.insert(0, PARKOUR_ROOT)

from training.train import ENV_REGISTRY
from models.actor_critic import ActorCritic
from models.latent_actor_critic import LatentActorCritic
from algos.dreamer import Dreamer
from algos.dreamer_video import DreamerVideo
from envs.env_client import EnvClient
from visualization.live_server import LiveServer
from visualization.live_frames import make_frame_builder, annotate

MOCK_STEP_DT = 0.15  # Malmo's ~6.6 Hz step rate


def parse_args():
    p = argparse.ArgumentParser(description="Live split-screen: Minecraft | world model")
    p.add_argument("--env", default="hunting", choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--checkpoint", default=None, help="dreamer / dreamer_video .pt")
    p.add_argument("--algo", default="auto", choices=["auto", "dreamer", "dreamer_video"],
                   help="auto = detect from checkpoint keys")
    p.add_argument("--port", type=int, default=9999, help="env server TCP port")
    p.add_argument("--http-port", type=int, default=8765, help="page port (default 8765)")
    p.add_argument("--greedy", action="store_true", help="argmax actions (default: sample)")
    p.add_argument("--third-person", action="store_true",
                   help="press F5+F1 in the Minecraft window (needs utils/send_mc_keys)")
    p.add_argument("--mock", action="store_true", help="no Malmo: synthetic obs + camera")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = forever)")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def detect_algo(args):
    if args.algo != "auto":
        return args.algo
    if not args.checkpoint:
        return "dreamer_video" if getattr(ENV_REGISTRY[args.env][1], "VIDEO_ENABLED", False) else "dreamer"
    keys = torch.load(args.checkpoint, map_location="cpu", weights_only=False).keys()
    if "video_world_model_state" in keys:
        return "dreamer_video"
    if "world_model_state" in keys:
        return "dreamer"
    raise SystemExit("'{0}' has no world-model state — not a dreamer / dreamer_video checkpoint."
                     .format(args.checkpoint))


# ── mock environment (no Malmo) ─────────────────────────────────────────────
class MockEnv:
    """Bounded random-walk obs + a synthetic 'camera' so the page can be built
    and tested without Minecraft. Mirrors EnvClient's reset/step/last_frame."""

    def __init__(self, cfg, video):
        self.cfg, self.video = cfg, video
        self.size = cfg.INPUT_SIZE
        self.t = 0
        self.last_frame = None
        self._obs = None

    def _camera(self, w, h):
        from PIL import Image, ImageDraw
        x = np.linspace(0, 1, w)[None, :, None]
        y = np.linspace(0, 1, h)[:, None, None]
        phase = self.t * 0.08
        rgb = np.concatenate([
            0.35 + 0.25 * np.sin(6 * x + phase) + 0 * y,
            0.55 + 0.2 * y + 0 * x,
            0.3 + 0.2 * np.cos(4 * y - phase) + 0 * x,
        ], axis=2)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        px = int((0.5 + 0.35 * np.sin(phase * 0.7)) * w)
        d.rectangle([px - w // 20, int(h * 0.55), px + w // 20, int(h * 0.7)], fill=(240, 160, 170))
        d.text((8, 8), "MOCK  step {0}".format(self.t), fill=(255, 255, 255))
        return np.asarray(img)

    def _obs_tuple(self):
        if self.video:
            frame = self._camera(self.cfg.VIDEO_WIDTH, self.cfg.VIDEO_HEIGHT)
            self.last_frame = frame
            return self._obs.copy(), frame
        self.last_frame = self._camera(640, 360)
        return self._obs.copy()

    def reset(self):
        self.t = 0
        self._obs = np.zeros(self.size, dtype=np.float32)
        return self._obs_tuple()

    def step(self, action):
        self.t += 1
        o = self._obs + np.random.randn(self.size).astype(np.float32) * 0.2
        self._obs = np.clip(o, -2.0, 2.0)
        done = self.t >= 120
        info = {"outcome": "timeout" if done else "running", "steps": self.t,
                "pos": (0, 0, 0), "action": self.cfg.ACTIONS[int(action)][0]}
        return self._obs_tuple(), float(np.random.randn() * 0.1), done, info

    def close(self):
        pass


def build_agent(cfg, algo, checkpoint):
    if algo == "dreamer_video":
        agent = DreamerVideo(LatentActorCritic(cfg), cfg)
    else:
        agent = Dreamer(ActorCritic(cfg), cfg)
    if checkpoint:
        agent.load(checkpoint)
    agent.model.eval()
    agent.world_model.eval()
    return agent


def run(args):
    cfg = ENV_REGISTRY[args.env][1]
    algo = detect_algo(args)
    video = algo == "dreamer_video"
    if video and not getattr(cfg, "VIDEO_ENABLED", False):
        raise SystemExit("dreamer_video needs a video env (e.g. --env hunting_video)")
    if not args.mock and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --mock")

    agent = build_agent(cfg, algo, args.checkpoint)
    builder = make_frame_builder(agent, cfg, args.env)

    server = LiveServer(args.http_port)
    server.publish_schema(builder.schema())
    print("\nLive split-screen:", server.url, " ({0}, {1})\n".format(algo, "MOCK" if args.mock else "Malmo"))
    if not args.no_browser:
        try:
            webbrowser.open(server.url)
        except Exception:
            pass

    if args.mock:
        env = MockEnv(cfg, video)
    else:
        env = EnvClient(cfg.INPUT_SIZE, port=args.port, video=video,
                        frame_shape=(cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS))

    total, episode, warned = 0, 0, False
    try:
        while True:
            episode += 1
            server.publish_status("reset", "starting episode {0}…".format(episode))
            obs = env.reset()
            if video:
                agent.reset_eval_state()
            if args.third_person and episode == 1 and not args.mock:
                from utils.mc_demo_camera import apply_demo_camera
                apply_demo_camera()
            server.publish_status("eval", "")
            done, t, ep_reward, last_r = False, 0, 0.0, None
            while not done:
                t0 = time.time()
                action = agent.select_action(obs, greedy=args.greedy)
                if server.has_clients:
                    cam = env.last_frame
                    if cam is None and not video and not warned:
                        warned = True
                        print("No camera frames arriving — start env_server with MALMO_RECORD_VIDEO=1.")
                    frame = builder.build(obs, cam=cam, mode="eval")
                    server.publish_frame(annotate(frame, cfg, action, episode, t, ep_reward, "eval",
                                                  last_reward=last_r))
                obs, last_r, done, info = env.step(action)
                ep_reward += last_r
                t += 1
                total += 1
                if done:
                    server.publish_stats("episode", {
                        "episode": episode, "return": round(float(ep_reward), 3),
                        "steps": t, "outcome": info.get("outcome", "?"), "env": args.env})
                    server.publish_status("done", "episode {0}: {1} · reward {2:.2f}".format(
                        episode, info.get("outcome", "?"), ep_reward))
                    print("  Ep {0:>3} | steps:{1:>3} | reward:{2:>7.2f} | {3}".format(
                        episode, t, ep_reward, info.get("outcome", "?")))
                if args.max_steps and total >= args.max_steps:
                    return
                if args.mock:
                    time.sleep(max(0.0, MOCK_STEP_DT - (time.time() - t0)))
    finally:
        env.close()
        server.close()


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
