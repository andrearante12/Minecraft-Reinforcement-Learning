"""
utils/replay_trajectory.py
---------------------------
Replay a scripted trajectory in Minecraft for visual testing.

Loads a trajectory .txt file (action names, one per line), connects to an env
server, and executes each action so you can watch the result in-game.

Usage:
    conda activate train_env
    python Malmo/rl/utils/replay_trajectory.py --env bridging --port 10002 --trajectory trajectories/bridging.txt

Trajectory file format:
    # comments are ignored
    sneak_backward          # one action per line
    sneak_backward x 3      # repeat syntax
    sneak_place
"""

import sys
import os
import argparse
import time
import functools

print = functools.partial(print, flush=True)

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from envs.env_client import EnvClient
from training.configs.simple_jump_cfg     import SimpleJumpCFG
from training.configs.one_block_gap_cfg   import OneBlockGapCFG
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG
from training.configs.bridging_cfg        import BridgingCFG

ENV_CONFIGS = {
    "simple_jump":     SimpleJumpCFG,
    "one_block_gap":   OneBlockGapCFG,
    "three_block_gap": ThreeBlockGapCFG,
    "bridging":        BridgingCFG,
}


def parse_trajectory(path, action_names):
    """Load trajectory .txt file → list of action indices."""
    name_to_idx = {name: i for i, name in enumerate(action_names)}
    actions = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if " x " in line:
                parts = line.split(" x ", 1)
                name = parts[0].strip()
                try:
                    count = int(parts[1].strip())
                except ValueError:
                    print("ERROR: line {0}: invalid repeat count '{1}'".format(lineno, parts[1].strip()))
                    sys.exit(1)
            else:
                name, count = line, 1
            if name not in name_to_idx:
                print("ERROR: line {0}: unknown action '{1}'".format(lineno, name))
                print("Valid actions: {0}".format(", ".join(action_names)))
                sys.exit(1)
            actions.extend([name_to_idx[name]] * count)
    return actions


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a scripted trajectory in Minecraft")
    parser.add_argument("--env", type=str, required=True, choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--trajectory", type=str, required=True,
                        help="Path to trajectory .txt file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]
    action_names = [a[0] for a in cfg.ACTIONS]

    if not os.path.exists(args.trajectory):
        print("ERROR: Trajectory file not found: {0}".format(args.trajectory))
        sys.exit(1)

    actions = parse_trajectory(args.trajectory, action_names)

    print("=" * 60)
    print("Trajectory Replay")
    print("=" * 60)
    print("Environment:  {0}".format(args.env))
    print("Trajectory:   {0}".format(args.trajectory))
    print("Steps:        {0}".format(len(actions)))
    print("Speed:        {0}x".format(args.speed))
    print("=" * 60)
    print()
    print("Actions:")
    for i, a in enumerate(actions):
        print("  {0:>4}: {1}".format(i + 1, action_names[a]))
    print()

    tick_delay = 0.05 / max(args.speed, 0.01)
    env = EnvClient(cfg.INPUT_SIZE, port=args.port)

    try:
        obs = env.reset()
        total_reward = 0.0
        print("--- Replay ---")
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            total_reward += reward
            outcome = ""
            if done and info:
                outcome = " | outcome: {0}".format(info.get("outcome", "unknown"))
            print("  step:{0:>4} | {1:<18} | reward:{2:>7.2f} | total:{3:>7.2f}{4}".format(
                i + 1, action_names[action], reward, total_reward, outcome))
            if done:
                break
            time.sleep(tick_delay)
        print()
        print("Total reward: {0:.2f}".format(total_reward))

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
