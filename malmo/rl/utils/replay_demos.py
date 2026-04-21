"""
utils/replay_demos.py
---------------------
Replay recorded demonstrations through a running env server.

Connects to an env server via EnvClient, loads a demo JSON file, and
feeds the stored actions back step-by-step so you can watch the replay
in the Minecraft client.

Usage:
    conda activate train_env
    python Malmo/rl/utils/replay_demos.py --env bridging --port 10002
    python Malmo/rl/utils/replay_demos.py --env bridging --port 10002 --episode 0 --speed 0.5
"""

import sys
import os
import json
import argparse
import time
import functools

# Force unbuffered stdout so logs appear in real time
print = functools.partial(print, flush=True)

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from envs.env_client import EnvClient

# Degrees the camera rotates per single discrete action.
# Derived from STEP_DURATION=0.15s × Malmo default turn speed (180°/s yaw, 90°/s pitch).
CAMERA_YAW_DEGS_PER_ACTION   = 27.0
CAMERA_PITCH_DEGS_PER_ACTION = 13.5

# Environment registry — only need configs for INPUT_SIZE and action names
from training.configs.simple_jump_cfg       import SimpleJumpCFG
from training.configs.one_block_gap_cfg     import OneBlockGapCFG
from training.configs.three_block_gap_cfg   import ThreeBlockGapCFG
from training.configs.diagonal_small_cfg    import DiagonalSmallCFG
from training.configs.diagonal_medium_cfg   import DiagonalMediumCFG
from training.configs.vertical_small_cfg    import VerticalSmallCFG
from training.configs.multi_jump_course_cfg import MultiJumpCourseCFG
from training.configs.multi_jump_branch_cfg import MultiJumpBranchCFG
from training.configs.bridging_cfg          import BridgingCFG

ENV_CONFIGS = {
    "simple_jump":       SimpleJumpCFG,
    "one_block_gap":     OneBlockGapCFG,
    "three_block_gap":   ThreeBlockGapCFG,
    "diagonal_small":    DiagonalSmallCFG,
    "diagonal_medium":   DiagonalMediumCFG,
    "vertical_small":    VerticalSmallCFG,
    "multi_jump_course": MultiJumpCourseCFG,
    "multi_jump_branch": MultiJumpBranchCFG,
    "bridging":          BridgingCFG,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Replay recorded demonstrations")
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_CONFIGS.keys()),
                        help="Environment to replay in")
    parser.add_argument("--port", type=int, default=9999,
                        help="Env server TCP port (default: 9999)")
    parser.add_argument("--demo-path", type=str, default=None,
                        help="Path to demo JSON (default: demos/<env>.json)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Replay only this episode index (default: all)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    return parser.parse_args()


def replay_episode(env, episode_data, episode_idx, action_names, tick_delay):
    """Replay a single episode through the env server.

    Returns:
        (total_reward, replay_outcome, recorded_outcome)
    """
    recorded_outcome = episode_data.get("outcome", "unknown")
    steps = episode_data["steps"]

    print("Episode {0} — {1} steps, recorded outcome: {2}".format(
        episode_idx, len(steps), recorded_outcome))
    print("-" * 60)

    obs = env.reset()
    total_reward = 0.0
    replay_outcome = "unknown"

    for i, step_data in enumerate(steps):
        action = step_data["action"]
        action_name = action_names[action] if action < len(action_names) else "action_{0}".format(action)

        yaw_delta   = step_data.get("yaw_delta")
        pitch_delta = step_data.get("pitch_delta")

        if yaw_delta is not None:
            n = round(yaw_delta / CAMERA_YAW_DEGS_PER_ACTION)
            if n == 0:
                continue
            for _ in range(n):
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
                time.sleep(tick_delay)
        elif pitch_delta is not None:
            n = round(pitch_delta / CAMERA_PITCH_DEGS_PER_ACTION)
            if n == 0:
                continue
            for _ in range(n):
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
                time.sleep(tick_delay)
        else:
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print("  step:{0:>4} | action:{1:<18} | reward:{2:>7.2f} | total:{3:>7.2f}".format(
            i + 1, action_name, reward, total_reward))

        if done:
            replay_outcome = info.get("outcome", "unknown") if info else "unknown"
            print("  >>> Episode ended early at step {0}/{1}: {2}".format(
                i + 1, len(steps), replay_outcome))
            break

        time.sleep(tick_delay)
    else:
        # All steps replayed without env signaling done
        replay_outcome = "completed_all_steps"

    print("-" * 60)
    print("  Total reward: {0:.2f}".format(total_reward))
    print("  Replay outcome:   {0}".format(replay_outcome))
    print("  Recorded outcome: {0}".format(recorded_outcome))
    if replay_outcome != recorded_outcome and replay_outcome != "completed_all_steps":
        print("  WARNING: replay outcome differs from recording (Minecraft non-determinism)")
    print()

    return total_reward, replay_outcome, recorded_outcome


def replay():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]
    action_names = [a[0] for a in cfg.ACTIONS]

    demo_path = args.demo_path or os.path.join(
        PARKOUR_ROOT, "..", "demos", "{0}.json".format(args.env))

    if not os.path.exists(demo_path):
        print("ERROR: Demo file not found: {0}".format(demo_path))
        sys.exit(1)

    with open(demo_path, "r") as f:
        data = json.load(f)

    episodes = data["episodes"]
    print("=" * 60)
    print("Demo Replay")
    print("=" * 60)
    print("Environment:  {0}".format(args.env))
    print("Demo file:    {0}".format(demo_path))
    print("Episodes:     {0}".format(len(episodes)))
    print("Speed:        {0}x".format(args.speed))
    print("=" * 60)
    print()

    if args.episode is not None:
        if args.episode < 0 or args.episode >= len(episodes):
            print("ERROR: Episode index {0} out of range (0-{1})".format(
                args.episode, len(episodes) - 1))
            sys.exit(1)
        to_replay = [(args.episode, episodes[args.episode])]
    else:
        to_replay = list(enumerate(episodes))

    base_tick = 0.05  # 20Hz, matching record_demos.py
    tick_delay = base_tick / max(args.speed, 0.01)

    env = EnvClient(cfg.INPUT_SIZE, port=args.port)

    try:
        for idx, (ep_idx, ep_data) in enumerate(to_replay):
            replay_episode(env, ep_data, ep_idx, action_names, tick_delay)

            # Pause between episodes if more remain
            if idx < len(to_replay) - 1:
                input("Press Enter for next episode (or Ctrl+C to quit)...")
                print()

    except KeyboardInterrupt:
        print("\nReplay interrupted.")

    finally:
        env.close()

    print("Replay complete.")


if __name__ == "__main__":
    replay()
