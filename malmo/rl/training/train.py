"""
training/train.py
-----------------
Generic training entrypoint. Select environment and algorithm via flags.

Requires env_server.py to be running first:
    Terminal 1: conda activate malmo && python parkour/envs/env_server.py --env simple_jump
    Terminal 2: conda activate train_env && python parkour/training/train.py --env simple_jump

Multi-env training (N=2 example):
    Terminal 1: python parkour/envs/env_server.py --env simple_jump --port 9999
    Terminal 2: python parkour/envs/env_server.py --env simple_jump --port 10000
    Terminal 3: python parkour/training/train.py --env simple_jump --algo ppo --num-envs 2

Usage:
    python parkour/training/train.py --env simple_jump --algo ppo
    python parkour/training/train.py --env three_block_gap --algo dqn
    python parkour/training/train.py --env simple_jump --algo ppo --checkpoint checkpoints/ppo_simple_jump_ep500.pt
"""

import sys
import os
import argparse
import numpy as np
import torch

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from training.configs.one_block_gap_cfg import OneBlockGapCFG
from models.actor_critic import ActorCritic
from models.latent_actor_critic import LatentActorCritic
from envs.env_client  import EnvClient
from utils.logger     import Logger
from training.curriculum import CurriculumScheduler

# ── Algorithm registry ────────────────────────────────────────────────────────
from algos.ppo import PPO
from algos.dqn import DQN
from algos.behavioral_cloning import BehavioralCloning
from algos.dreamer import Dreamer
from algos.dreamer_video import DreamerVideo

ALGO_REGISTRY = {
    "ppo":           PPO,
    "dqn":           DQN,
    "bc":            BehavioralCloning,
    "dreamer":       Dreamer,
    "dreamer_video": DreamerVideo,
}

# ── Environment registry ──────────────────────────────────────────────────────
# Tuples of (EnvClass, CfgClass) — EnvClass is not used directly by the client,
# but the scheduler needs the full registry for validation.
from training.configs.simple_jump_cfg       import SimpleJumpCFG
from training.configs.three_block_gap_cfg   import ThreeBlockGapCFG
from training.configs.diagonal_small_cfg    import DiagonalSmallCFG
from training.configs.diagonal_medium_cfg   import DiagonalMediumCFG
from training.configs.vertical_small_cfg      import VerticalSmallCFG
from training.configs.multi_jump_course_cfg   import MultiJumpCourseCFG
from training.configs.multi_jump_branch_cfg   import MultiJumpBranchCFG
from training.configs.bridging_cfg            import BridgingCFG
from training.configs.bridging_1block_cfg     import Bridging1BlockCFG
from training.configs.bridging_2block_cfg     import Bridging2BlockCFG
from training.configs.bridging_3block_cfg     import Bridging3BlockCFG
from training.configs.bridging_4block_cfg     import Bridging4BlockCFG
from training.configs.bridging_straight_cfg   import BridgingStraightCFG
from training.configs.hunting_cfg              import HuntingCFG
from training.configs.hunting_wild_cfg         import HuntingWildCFG
from training.configs.hunting_video_cfg        import HuntingVideoCFG
from training.configs.hunting_vision_cfg       import HuntingVisionCFG

ENV_REGISTRY = {
    "one_block_gap":       (None, OneBlockGapCFG),
    "simple_jump":         (None, SimpleJumpCFG),
    "three_block_gap":     (None, ThreeBlockGapCFG),
    "diagonal_small":      (None, DiagonalSmallCFG),
    "diagonal_medium":     (None, DiagonalMediumCFG),
    "vertical_small":      (None, VerticalSmallCFG),
    "multi_jump_course":   (None, MultiJumpCourseCFG),
    "multi_jump_branch":   (None, MultiJumpBranchCFG),
    "bridging":            (None, BridgingCFG),
    "bridging_1block":     (None, Bridging1BlockCFG),
    "bridging_2block":     (None, Bridging2BlockCFG),
    "bridging_3block":     (None, Bridging3BlockCFG),
    "bridging_4block":     (None, Bridging4BlockCFG),
    "bridging_straight":   (None, BridgingStraightCFG),
    "bridging_5block":     (None, BridgingCFG),
    "hunting":             (None, HuntingCFG),
    "hunting_wild":        (None, HuntingWildCFG),
    "hunting_video":       (None, HuntingVideoCFG),
    "hunting_vision":      (None, HuntingVisionCFG),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        type=str, default="simple_jump",
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment to train in (default: simple_jump)")
    parser.add_argument("--algo",       type=str, default="ppo",
                        choices=list(ALGO_REGISTRY.keys()),
                        help="RL algorithm to use (default: ppo)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num-envs",   type=int, default=1,
                        help="Number of parallel env servers (default: 1)")
    parser.add_argument("--base-port",  type=int, default=9999,
                        help="Port of first env server; others at base+1, base+2, ... (default: 9999)")
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Path to curriculum JSON file (overrides --env for scheduling)")
    parser.add_argument("--demo-path", type=str, default=None,
                        help="Path to demo JSON file (required for --algo bc); for "
                             "--algo dreamer_video, a frames demo directory (utils/demo_io.py)")
    parser.add_argument("--pretrain-wm-steps", type=int, default=0,
                        help="dreamer_video: fit the world model on --demo-path demos for "
                             "this many steps before the first live step (default 0 = off)")
    parser.add_argument("--viz-port", type=int, default=None,
                        help="Serve the live split-screen page (Minecraft | model predictions) "
                             "on this port, e.g. 8765. Off by default.")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Save an mp4 of every successful episode (killed, or reward >= "
                             "--record-min-reward) of env 0 into this directory. Needs a video env.")
    parser.add_argument("--record-max", type=int, default=15,
                        help="Stop recording after this many clips (default 15)")
    parser.add_argument("--record-min-reward", type=float, default=150.0,
                        help="Also record non-kill episodes scoring at least this (default 150)")
    parser.add_argument("--viz-env", type=int, default=0,
                        help="Which parallel env to stream with --viz-port (default 0)")
    parser.add_argument("--viz-third-person", action="store_true",
                        help="With --viz-port: press F5+F1 in the Minecraft window at startup")
    return parser.parse_args()


def print_header(env_name, algo_name, cfg, num_envs=1, total_episodes=None):
    print("=" * 60)
    print("Malmo RL Training")
    print("=" * 60)
    print("Environment:    ", env_name)
    print("Algorithm:      ", algo_name.upper())
    print("Device:         ", "cuda" if torch.cuda.is_available() else "cpu")
    print("Num envs:       ", num_envs)
    print("Total episodes: ", total_episodes if total_episodes is not None else cfg.TOTAL_EPISODES)
    print("Obs size:       ", cfg.INPUT_SIZE)
    print("N actions:      ", cfg.N_ACTIONS)
    print("Architecture:    ", "multi-stream ({0}/{1}/{2}) -> {3}".format(
        cfg.PROPRIO_HIDDEN, cfg.GOAL_HIDDEN, cfg.VOXEL_HIDDEN, cfg.HEAD_HIDDEN))
    if algo_name == "ppo":
        print("N steps/update: ", cfg.N_STEPS)
        if num_envs > 1:
            print("Steps per env:  ", cfg.N_STEPS // num_envs)
        print("Batch size:     ", cfg.BATCH_SIZE)
        print("LR:             ", cfg.LR)
        print("Gamma:          ", cfg.GAMMA)
        print("Clip eps:       ", cfg.CLIP_EPS)
        print("Entropy coef:   ", cfg.ENTROPY_COEF)
    elif algo_name == "dqn":
        print("Buffer capacity:", cfg.BUFFER_CAPACITY)
        print("Batch size:     ", cfg.BATCH_SIZE)
        print("LR:             ", cfg.LR)
        print("Gamma:          ", cfg.GAMMA)
        print("Epsilon start:  ", cfg.EPSILON_START)
        print("Epsilon end:    ", cfg.EPSILON_END)
        print("Target update:  ", cfg.TARGET_UPDATE_FREQ)
    elif algo_name == "bc":
        print("Demo path:      ", cfg.DEMO_PATH)
        print("BC epochs:      ", cfg.BC_EPOCHS)
        print("BC batch size:  ", cfg.BC_BATCH_SIZE)
        print("LR:             ", cfg.LR)
    elif algo_name == "dreamer_video":
        print("Video:          ", "{0}x{1}x{2}".format(cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS))
        if getattr(cfg, "VWM_INPUT", "pixels") == "features":
            print("Frozen encoder: ", "{0} (pool {1}x{1})".format(cfg.VIS_ENCODER, cfg.VIS_POOL_GRID))
        if cfg.DEMO_PATH:
            print("Demos:          ", cfg.DEMO_PATH)
        print("RSSM:           ", "deter={0} stoch={1}".format(cfg.RSSM_DETER, cfg.WM_LATENT_DIM))
        print("Seq len:        ", cfg.WM_SEQ_LEN)
        print("Seq buffer cap: ", cfg.SEQ_BUFFER_CAPACITY)
        print("Imag horizon:   ", cfg.IMAG_HORIZON)
    print("=" * 60)
    print()


class EpisodeRecorder:
    """Buffers env-0 camera frames; writes an mp4 (ffmpeg) when an episode succeeded."""

    def __init__(self, out_dir, max_clips, min_reward, fps=10, scale=3):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.max_clips, self.min_reward = out_dir, max_clips, min_reward
        self.fps, self.scale, self.saved, self.frames = fps, scale, 0, []

    def add(self, frame):
        if frame is not None and self.saved < self.max_clips:
            self.frames.append(np.array(frame, dtype=np.uint8))

    def finish(self, episode, outcome, reward, steps):
        frames, self.frames = self.frames, []
        if (not frames or self.saved >= self.max_clips
                or not (outcome == "killed" or reward >= self.min_reward)):
            return
        import subprocess
        h, w = frames[0].shape[:2]
        path = os.path.join(self.out_dir, "ep{0}_{1}_r{2:.0f}_s{3}.mp4".format(
            episode, outcome, reward, steps))
        try:
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "{0}x{1}".format(w, h), "-r", str(self.fps), "-i", "-",
                 "-vf", "scale={0}:{1}:flags=neighbor".format(w * self.scale, h * self.scale),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                stdin=subprocess.PIPE)
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
            proc.wait(timeout=120)
            self.saved += 1
            print("  recorded {0} ({1} frames)".format(path, len(frames)))
        except Exception as e:
            print("  recording failed: {0}".format(e))


def prune_checkpoints(cfg, prefix):
    """Keep only the newest cfg.KEEP_CHECKPOINTS periodic checkpoints of this run
    (files `<prefix>_ep<N>.pt`; *_interrupted and names in KEEP_CHECKPOINT_EPISODES
    are never removed). Off unless the config sets KEEP_CHECKPOINTS."""
    keep = getattr(cfg, "KEEP_CHECKPOINTS", None)
    if not keep:
        return
    import re
    pinned = set(getattr(cfg, "KEEP_CHECKPOINT_EPISODES", ()))
    pat = re.compile(re.escape(prefix) + r"_ep(\d+)\.pt$")
    found = []
    for name in os.listdir(cfg.CHECKPOINT_DIR):
        m = pat.match(name)
        if m and int(m.group(1)) not in pinned:
            found.append((int(m.group(1)), name))
    for _, name in sorted(found)[:-keep]:
        os.remove(os.path.join(cfg.CHECKPOINT_DIR, name))


def print_update(losses):
    metrics = " | ".join("{0}:{1:.4f}".format(k, v) for k, v in losses.items())
    print("  [update] {0}".format(metrics))


def train():
    args    = parse_args()
    n_envs  = args.num_envs

    # Build curriculum scheduler
    if args.curriculum:
        scheduler = CurriculumScheduler.from_json(args.curriculum, ENV_REGISTRY)
        _, cfg = ENV_REGISTRY[scheduler.all_env_names()[0]]
        total_episodes = scheduler.total_episodes()
        run_name = "curriculum_{0}".format(args.algo)
    else:
        _, cfg = ENV_REGISTRY[args.env]
        total_episodes = cfg.TOTAL_EPISODES
        scheduler = CurriculumScheduler.single_env(args.env, total_episodes, ENV_REGISTRY)
        run_name = "{0}_{1}".format(args.env, args.algo)

    # Pass demo path to config for BC
    if args.demo_path:
        cfg.DEMO_PATH = args.demo_path

    # Validate N_STEPS divisibility for PPO
    if args.algo == "ppo" and cfg.N_STEPS % n_envs != 0:
        print("ERROR: N_STEPS ({0}) must be divisible by --num-envs ({1})".format(
            cfg.N_STEPS, n_envs))
        sys.exit(1)

    # dreamer_video <-> video env pairing
    is_video = (args.algo == "dreamer_video")
    env_names = (scheduler.all_env_names() if args.curriculum else [args.env])
    env_video_flags = [getattr(ENV_REGISTRY[name][1], "VIDEO_ENABLED", False) for name in env_names]
    if is_video and not all(env_video_flags):
        print("ERROR: --algo dreamer_video requires a video-enabled env (e.g. --env hunting_video). "
              "Got env(s) {0} with VIDEO_ENABLED={1}".format(env_names, env_video_flags))
        sys.exit(1)
    if not is_video and any(env_video_flags):
        print("WARNING: env(s) {0} have VIDEO_ENABLED=True but --algo {1} doesn't use video — "
              "frames will be shipped over the wire and discarded (wasted bandwidth).".format(
                  env_names, args.algo))

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Create N env clients, each connecting to base_port+i
    base_port = args.base_port
    envs   = [EnvClient(cfg.INPUT_SIZE, port=base_port + i, video=is_video,
                        frame_shape=(cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS))
              for i in range(n_envs)]
    model  = LatentActorCritic(cfg) if is_video else ActorCritic(cfg)
    agent  = ALGO_REGISTRY[args.algo](model, cfg, n_envs=n_envs)
    logger = Logger(cfg.LOG_DIR, run_name)

    if args.curriculum:
        if scheduler.mode == "adaptive":
            env_label = "adaptive ({0} stages: {1})".format(
                len(scheduler.stages),
                " -> ".join(s["env"] for s in scheduler.stages))
        else:
            env_label = args.curriculum
    else:
        env_label = args.env
    print_header(env_label, args.algo, cfg, num_envs=n_envs, total_episodes=total_episodes)

    start_episode = 1
    if args.checkpoint:
        agent.load(args.checkpoint)
        try:
            ep_str = args.checkpoint.split("ep")[-1].split(".")[0]
            # Strip non-digit suffixes like "_interrupted"
            ep_num = ""
            for ch in ep_str:
                if ch.isdigit():
                    ep_num += ch
                else:
                    break
            start_episode = int(ep_num) + 1
        except Exception:
            pass
        # Restore adaptive curriculum state if sidecar exists
        curriculum_path = args.checkpoint.replace(".pt", "_curriculum.pt")
        if os.path.exists(curriculum_path):
            scheduler.load_state_dict(torch.load(curriculum_path, weights_only=False))
            print("Restored curriculum state from {0}".format(curriculum_path))
        print("Resuming from episode {0}".format(start_episode))
        print()

    if args.pretrain_wm_steps > 0:
        if hasattr(agent, "pretrain_world_model"):
            agent.pretrain_world_model(args.pretrain_wm_steps)
            print()
        else:
            print("WARNING: --pretrain-wm-steps is only supported by --algo dreamer_video; ignored")

    # Determine initial env for each slot
    initial_env = scheduler.env_for_episode(start_episode)
    current_envs = [initial_env] * n_envs
    logger.init_trajectory(initial_env, cfg)

    # Per-env state tracking. Video obs are (vec, frame) tuples, so obs_all is a
    # plain Python list rather than a preallocated ndarray in that case.
    obs_all    = [None] * n_envs if is_video else np.zeros((n_envs, cfg.INPUT_SIZE), dtype=np.float32)
    ep_rewards = np.zeros(n_envs, dtype=np.float64)
    ep_steps   = np.zeros(n_envs, dtype=np.int64)
    ep_outcome = ["timeout"] * n_envs

    for i, env in enumerate(envs):
        env.switch_env(initial_env)
        obs_all[i] = env.reset()

    # Live split-screen viewer (visualization/live_split.py has the playback twin).
    viz = viz_builder = None
    if args.viz_port:
        from visualization.live_server import LiveServer
        from visualization.live_frames import make_frame_builder, annotate
        viz_k = min(max(args.viz_env, 0), n_envs - 1)
        viz_builder = make_frame_builder(agent, cfg, current_envs[viz_k])
        viz = LiveServer(args.viz_port)
        viz.publish_schema(viz_builder.schema())
        viz.publish_status("collect", "")
        print("Live view: {0}  (env {1})".format(viz.url, viz_k))
        if not is_video and envs[viz_k].last_frame is None:
            print("  No camera frames from env server — restart it with MALMO_RECORD_VIDEO=1 "
                  "to see Minecraft in the live view.")
        if args.viz_third_person:
            from utils.mc_demo_camera import apply_demo_camera
            apply_demo_camera()
        print()

    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)   # background shells ignore it
    recorder = None
    if args.record_dir:
        recorder = EpisodeRecorder(args.record_dir, args.record_max, args.record_min_reward)
        print("Recording successful episodes of env 0 to {0} (max {1})".format(
            args.record_dir, args.record_max))

    episode      = start_episode
    update_count = 0

    print("Starting training... (Ctrl+C to stop and save)")
    print()

    try:
        while episode <= total_episodes and not scheduler.is_complete():

            viz_frame = None
            if viz is not None and viz.has_clients:
                # Predictions on the PRE-step obs, i.e. the frame the agent is acting on.
                try:
                    viz_frame = viz_builder.build(obs_all[viz_k], cam=envs[viz_k].last_frame,
                                                  env_idx=viz_k, mode="train")
                except Exception as e:  # never let the viewer kill a training run
                    print("  live view: frame build failed ({0}); disabling".format(e))
                    viz = None

            next_obs_all, rewards, dones, infos = agent.collect_steps(envs, obs_all)

            if viz_frame is not None:
                act_names = [a[0] for a in cfg.ACTIONS]
                act_name = infos[viz_k].get("action")
                act_idx = act_names.index(act_name) if act_name in act_names else viz_frame["action"]
                viz.publish_frame(annotate(
                    viz_frame, cfg, act_idx, episode, int(ep_steps[viz_k]),
                    ep_rewards[viz_k] + rewards[viz_k], "train", last_reward=rewards[viz_k],
                    outcome=infos[viz_k].get("outcome")))

            for i in range(n_envs):
                ep_rewards[i] += rewards[i]
                ep_steps[i]   += 1

                logger.log_step(
                    episode  = episode,
                    step     = int(ep_steps[i]) - 1,
                    info     = infos[i],
                    reward   = rewards[i],
                    done     = dones[i],
                    env_name = current_envs[i],
                )

                if recorder is not None and i == 0:
                    recorder.add(envs[0].last_frame)

                if dones[i]:
                    ep_outcome[i] = infos[i]["outcome"]

                if dones[i]:
                    if hasattr(agent, "set_progress"):
                        agent.set_progress(episode / total_episodes)

                    logger.log_episode(episode, ep_rewards[i], int(ep_steps[i]),
                                       ep_outcome[i], env_name=current_envs[i])
                    if viz is not None:
                        viz.publish_stats("episode", {
                            "episode": int(episode), "return": round(float(ep_rewards[i]), 3),
                            "steps": int(ep_steps[i]), "outcome": ep_outcome[i],
                            "env": current_envs[i]})
                    logger.print_summary(every=cfg.LOG_EVERY)

                    # Report outcome to adaptive scheduler (no-op for other modes)
                    scheduler.report_outcome(ep_outcome[i])

                    ep_line = "  Ep {0:>4} | env:{1} | steps:{2:>3} | reward:{3:>7.2f} | outcome:{4}".format(
                        episode, current_envs[i], int(ep_steps[i]), ep_rewards[i], ep_outcome[i])
                    if scheduler.mode == "adaptive":
                        ep_line += " | sr:{0:.0%}".format(scheduler.current_success_rate())
                    print(ep_line)
                    if recorder is not None and i == 0:
                        recorder.finish(episode, ep_outcome[0], float(ep_rewards[0]), int(ep_steps[0]))

                    # Switch env if curriculum says so
                    next_env = scheduler.env_for_episode(episode + 1)
                    if next_env != current_envs[i]:
                        envs[i].switch_env(next_env)
                        current_envs[i] = next_env

                    # Reset this env
                    try:
                        next_obs_all[i] = envs[i].reset()
                    except RuntimeError as e:
                        print("\nMalmo reset failed — saving checkpoint and exiting: {0}".format(e))
                        path = os.path.join(cfg.CHECKPOINT_DIR,
                                            "{0}_{1}_ep{2}_interrupted.pt".format(args.algo, run_name, episode))
                        agent.save(path)
                        curriculum_state = scheduler.state_dict()
                        if curriculum_state is not None:
                            torch.save(curriculum_state, path.replace(".pt", "_curriculum.pt"))
                        raise
                    ep_rewards[i]   = 0.0
                    ep_steps[i]     = 0
                    ep_outcome[i]   = "timeout"

                    if episode % cfg.SAVE_EVERY == 0:
                        path = os.path.join(cfg.CHECKPOINT_DIR,
                                            "{0}_{1}_ep{2}.pt".format(args.algo, run_name, episode))
                        agent.save(path)
                        prune_checkpoints(cfg, "{0}_{1}".format(args.algo, run_name))
                        curriculum_state = scheduler.state_dict()
                        if curriculum_state is not None:
                            torch.save(curriculum_state, path.replace(".pt", "_curriculum.pt"))

                    episode += 1
                    if episode > total_episodes or scheduler.is_complete():
                        break

            obs_all = next_obs_all

            if agent.buffer_full():
                if viz is not None:
                    viz.publish_status("update", "training world model + imagination (update {0})…".format(
                        update_count + 1))
                losses = agent.update(last_obs=obs_all)
                if viz is not None:
                    viz.publish_status("collect", "")
                    stats = {k: round(float(v), 6) for k, v in losses.items()
                             if isinstance(v, (int, float, np.floating))}
                    stats.update(update=update_count + 1, episode=int(episode),
                                 env_steps=int(getattr(agent, "steps_done", 0)),
                                 buffer=len(agent.buffer) if hasattr(agent, "buffer") else None)
                    viz.publish_stats("update", stats)
                update_count += 1
                logger.log_update(**losses)
                if update_count % 10 == 0:
                    print_update(losses)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        path = os.path.join(cfg.CHECKPOINT_DIR,
                            "{0}_{1}_ep{2}_interrupted.pt".format(args.algo, run_name, episode))
        agent.save(path)
        curriculum_state = scheduler.state_dict()
        if curriculum_state is not None:
            torch.save(curriculum_state, path.replace(".pt", "_curriculum.pt"))

    finally:
        if viz is not None:
            viz.close()
        for env in envs:
            env.close()
        logger.close()
        print("Done.")


if __name__ == "__main__":
    train()
