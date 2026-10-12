"""
Record a short WebM from a live Malmo env (video frames over the wire).

Requires:
  T1  Minecraft client
  T2  env_server with MALMO_RECORD_VIDEO=1
  T3  this script (train_env)

    conda run -n train_env python malmo/rl/utils/record_eval_video.py \\
        --env simple_jump --checkpoint malmo/rl/baselines/ppo_parkour_baseline.pt \\
        --out site/media/parkour.webm
"""

from __future__ import print_function

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from envs.env_client import EnvClient


def _encode(frames, out_path, fps):
    if not frames:
        raise RuntimeError("no frames to encode")
    tmp = tempfile.mkdtemp(prefix="malmorl_clip_")
    try:
        for i, fr in enumerate(frames):
            path = os.path.join(tmp, "f_{0:04d}.ppm".format(i))
            h, w = fr.shape[0], fr.shape[1]
            with open(path, "wb") as f:
                f.write("P6\n{0} {1}\n255\n".format(w, h).encode("ascii"))
                f.write(np.ascontiguousarray(fr).tobytes())
        poster = os.path.splitext(out_path)[0] + ".jpg"
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "f_%04d.ppm"),
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36",
            "-pix_fmt", "yuv420p", "-an", out_path,
        ])
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", os.path.join(tmp, "f_0000.ppm"),
            "-q:v", "4", poster,
        ])
        print("wrote", out_path, "({0} frames) and {1}".format(len(frames), poster))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _policy_parkour(args, cfg):
    from models.actor_critic import ActorCritic
    from algos.ppo import PPO
    model = ActorCritic(cfg)
    agent = PPO(model, cfg)
    agent.load(args.checkpoint)
    model.eval()

    def act(obs):
        return int(agent.select_action(obs, greedy=True)), None
    return act, None, None


def _policy_bridging(args, cfg):
    import torch
    from stable_baselines3 import PPO
    from visualization.activation_probe import build_sb3_mlp_probe
    from visualization.net_schema import (
        Sb3BridgingFrameBuilder,
        build_sb3_schema,
    )

    model = PPO.load(args.checkpoint)
    model.policy.eval()
    probe = build_sb3_mlp_probe(model.policy)

    def act(obs):
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            actions, _values, _log_prob = model.policy(
                obs_tensor, deterministic=True
            )
        action = int(actions.reshape(-1)[0].cpu().item())
        return action, probe.snapshot()

    replay = {
        "schema": build_sb3_schema(cfg, model.policy),
        "builder_cls": Sb3BridgingFrameBuilder,
        "policy": model.policy,
    }
    return act, probe, replay


def _policy_hunting(args, cfg):
    from utils.generate_hunting_demos import scripted_action

    def act(obs):
        return int(scripted_action(obs, cfg)), None
    return act, None, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--port", type=int, default=10002)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=None,
                   help="Output fps (default: 1 / environment STEP_DURATION)")
    p.add_argument("--json-out", default=None,
                   help="Paired replay JSON path (bridging defaults beside WebM)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--f5", action="store_true",
                   help="Tap F5 once after the first reset (third-person toggle)")
    p.add_argument("--no-f1", action="store_true",
                   help="Do not tap F1 (GUI already hidden)")
    p.add_argument("--camera-configured", action="store_true",
                   help="options.txt already sets third person + hidden GUI")
    p.add_argument("--camera-ready-file", default=None,
                   help="Wait after mission start until this manual-camera marker exists")
    p.add_argument("--manual-camera", action="store_true",
                   help="Wait for SIGUSR1 after the user confirms third-person view")
    return p.parse_args()


def main():
    args = parse_args()
    camera_confirmed = [False]
    if args.manual_camera:
        def _confirm_camera(_signum, _frame):
            camera_confirmed[0] = True
        signal.signal(signal.SIGUSR1, _confirm_camera)

    from training.train import ENV_REGISTRY
    if args.env not in ENV_REGISTRY:
        sys.exit("unknown env: " + args.env)
    _, cfg_cls = ENV_REGISTRY[args.env]
    cfg = cfg_cls

    if args.env.startswith("bridg"):
        act, probe, replay_spec = _policy_bridging(args, cfg)
        prefer = ("landed",)
    elif args.env.startswith("hunt"):
        act, probe, replay_spec = _policy_hunting(args, cfg)
        prefer = ("killed", "success")
    else:
        if not args.checkpoint:
            sys.exit("--checkpoint required for parkour")
        act, probe, replay_spec = _policy_parkour(args, cfg)
        prefer = ("success", "landed", "goal")

    env = EnvClient(
        cfg.INPUT_SIZE, port=args.port, video=True,
        frame_shape=(args.height, args.width, 3),
    )

    best_frames = []
    best_replay = []
    best_outcome = None
    best_reward = None
    best_score = -1e9
    camera_set = False
    try:
        for ep in range(1, args.episodes + 1):
            pair = env.reset()
            obs, frame = pair
            if not camera_set:
                try:
                    if args.manual_camera:
                        print(
                            "mission ready; manually select third-person, "
                            "then send SIGUSR1 to pid", os.getpid()
                        )
                        deadline = time.time() + 600.0
                        while not camera_confirmed[0]:
                            if time.time() >= deadline:
                                raise RuntimeError(
                                    "timed out waiting for manual camera"
                                )
                            time.sleep(0.2)
                        camera_ok = True
                    elif args.camera_ready_file:
                        print(
                            "mission ready; waiting for manual third-person "
                            "camera marker:", args.camera_ready_file
                        )
                        deadline = time.time() + 600.0
                        while not os.path.exists(args.camera_ready_file):
                            if time.time() >= deadline:
                                raise RuntimeError(
                                    "timed out waiting for manual camera"
                                )
                            time.sleep(0.2)
                        camera_ok = True
                    elif args.camera_configured:
                        camera_ok = True
                    else:
                        from utils.mc_demo_camera import apply_demo_camera
                        camera_ok = apply_demo_camera(
                            press_f5=args.f5, press_f1=not args.no_f1
                        )
                    if not camera_ok:
                        raise RuntimeError("Minecraft camera key injection failed")
                except Exception as exc:
                    raise RuntimeError("camera setup failed: {0}".format(exc))
                camera_set = True
                time.sleep(0.25)
                # drop the pre-F5 frame; next step is the real demo view
                settle_action, _ = act(obs)
                obs, reward, done, info, frame = _step(env, settle_action)
                if done:
                    print("ep {0}: ended during camera settle ({1})".format(
                        ep, info.get("outcome", "?")))
                    continue
            # Record only post-action frames so video frame i and replay item i
            # share one unambiguous policy decision.
            frames = []
            total = 0.0
            outcome = "timeout"
            done = False
            episode_replay = []
            frame_builder = None
            if replay_spec is not None:
                frame_builder = replay_spec["builder_cls"](
                    cfg, replay_spec["policy"]
                )
            while not done:
                policy_obs = obs.copy()
                action, snapshot = act(policy_obs)
                obs, reward, done, info, frame = _step(env, action)
                total += float(reward)
                frames.append(frame)
                if frame_builder is not None:
                    episode_replay.append(frame_builder.build(
                        snapshot=snapshot,
                        action=action,
                        obs=policy_obs,
                        reward=reward,
                        outcome=info.get("outcome", "running"),
                        frame_index=len(frames) - 1,
                        step_dt=cfg.STEP_DURATION,
                    ))
                if done:
                    outcome = info.get("outcome", "unknown")
            score = total
            if any(k in str(outcome).lower() for k in prefer):
                score += 1000.0
            print("ep {0}: steps={1} reward={2:.1f} outcome={3}".format(
                ep, len(frames), total, outcome))
            succeeded = any(k in str(outcome).lower() for k in prefer)
            if succeeded and score > best_score and len(frames) > 3:
                best_score = score
                best_frames = frames
                best_replay = episode_replay
                best_outcome = outcome
                best_reward = total
                # A successful demo is the terminal objective. Finalize it now
                # instead of risking its loss to a later Malmo mission reset.
                break
    finally:
        env.close()
        if probe is not None:
            probe.remove()

    if not best_frames:
        raise RuntimeError(
            "no successful {0} episode in {1} attempts; outputs unchanged".format(
                args.env, args.episodes
            )
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = args.fps or (1.0 / float(cfg.STEP_DURATION))
    _encode(best_frames, args.out, fps)

    if replay_spec is not None:
        json_out = args.json_out or os.path.splitext(args.out)[0] + "_replay.json"
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        payload = {
            "schema": replay_spec["schema"],
            "meta": {
                "synced": True,
                "env": args.env,
                "checkpoint": args.checkpoint,
                "outcome": best_outcome,
                "episode_reward": round(float(best_reward), 4),
                "step_duration": float(cfg.STEP_DURATION),
                "fps": float(fps),
                "frame_count": len(best_frames),
                "pairing": "pre-action policy state -> post-action video frame",
                "camera": "third_person" if (
                    args.f5 or args.camera_configured or args.camera_ready_file
                    or args.manual_camera
                ) else "configured",
                "hud_hidden": bool(
                    args.camera_configured or args.camera_ready_file
                    or args.manual_camera or not args.no_f1
                ),
            },
            "frames": best_replay,
        }
        with open(json_out, "w") as f:
            json.dump(payload, f)
        print("wrote", json_out, "({0} replay frames)".format(len(best_replay)))


def _step(env, action):
    """EnvClient.step returns (obs, r, done, info) or with video a tuple obs."""
    result = env.step(action)
    # video client: step still returns obs as (vec, frame) plus r, done, info
    obs, reward, done, info = result
    if isinstance(obs, tuple):
        vec, frame = obs
        return vec, reward, done, info, frame
    return obs, reward, done, info, np.zeros((360, 640, 3), dtype=np.uint8)


if __name__ == "__main__":
    main()
