"""
utils/record_trajectory.py
---------------------------
Execute a scripted trajectory and save the episode as a demo for BC training.

Loads a trajectory .txt file, runs it through the env server, captures real
observations at each step, and saves the episode to a demo JSON file in the
same format that record_demos.py produces (and that train_sb3.py expects).

Usage:
    conda activate train_env
    python Malmo/rl/utils/record_trajectory.py --env bridging --port 10002 --trajectory trajectories/bridging.txt
    python Malmo/rl/utils/record_trajectory.py --env bridging --port 10002 --trajectory trajectories/bridging.txt --output demos/bridging.json
    python Malmo/rl/utils/record_trajectory.py --env bridging --port 10002 --trajectory trajectories/bridging.txt --runs 10

Trajectory file format:
    # comments are ignored
    sneak_backward          # one action per line
    sneak_backward x 3      # repeat syntax
    sneak_place
"""

import sys
import os
import json
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
    parser = argparse.ArgumentParser(description="Record a scripted trajectory as a demo episode")
    parser.add_argument("--env", type=str, required=True, choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--trajectory", type=str, required=True,
                        help="Path to trajectory .txt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output demo JSON path (default: demos/<env>.json)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to run the trajectory and record (default: 1)")
    return parser.parse_args()


def run_trajectory(env, actions, action_names, run_idx):
    """Execute one run of the trajectory. Returns the episode dict."""
    obs = env.reset(force_reset=True)
    steps = []
    total_reward = 0.0
    outcome = "completed"

    print("Run {0} — {1} steps".format(run_idx + 1, len(actions)))
    print("-" * 50)

    for i, action in enumerate(actions):
        obs_prev = obs.copy()
        obs, reward, done, info = env.step(action)
        total_reward += reward

        step_record = {"obs": obs_prev.tolist(), "action": int(action)}
        steps.append(step_record)

        print("  step:{0:>4} | {1:<18} | reward:{2:>7.2f} | total:{3:>7.2f}".format(
            i + 1, action_names[action], reward, total_reward))

        if done:
            outcome = info.get("outcome", "unknown") if info else "unknown"
            print("  >>> Done at step {0}/{1}: {2}".format(i + 1, len(actions), outcome))
            break

    print("-" * 50)
    print("  Total reward: {0:.2f}  |  Outcome: {1}".format(total_reward, outcome))
    print()

    return {"outcome": outcome, "steps": steps}


def main():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]
    action_names = [a[0] for a in cfg.ACTIONS]

    if not os.path.exists(args.trajectory):
        print("ERROR: Trajectory file not found: {0}".format(args.trajectory))
        sys.exit(1)

    actions = parse_trajectory(args.trajectory, action_names)

    output_path = args.output or os.path.join(
        PARKOUR_ROOT, "..", "demos", "{0}.json".format(args.env))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load existing demos if file exists (append mode)
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        print("Loaded {0} existing episodes from {1}".format(len(data["episodes"]), output_path))
    else:
        data = {"env": args.env, "episodes": []}

    print("=" * 60)
    print("Trajectory Recorder")
    print("=" * 60)
    print("Environment:  {0}".format(args.env))
    print("Trajectory:   {0}".format(args.trajectory))
    print("Steps:        {0}".format(len(actions)))
    print("Runs:         {0}".format(args.runs))
    print("Output:       {0}".format(output_path))
    print("=" * 60)
    print()
    print("Actions to execute:")
    for i, a in enumerate(actions):
        print("  {0:>4}: {1}".format(i + 1, action_names[a]))
    print()

    env = EnvClient(cfg.INPUT_SIZE, port=args.port)

    try:
        for run_idx in range(args.runs):
            episode = run_trajectory(env, actions, action_names, run_idx)
            data["episodes"].append(episode)

            # Auto-save after each run
            with open(output_path, "w") as f:
                json.dump(data, f)
            print("  Saved {0} total episodes to {1}".format(len(data["episodes"]), output_path))
            print()

            if run_idx < args.runs - 1:
                input("Press Enter for next run (or Ctrl+C to quit)...")
                print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()

    total_steps = sum(len(ep["steps"]) for ep in data["episodes"])
    print("Recording complete.")
    print("  Episodes: {0}".format(len(data["episodes"])))
    print("  Total steps: {0}".format(total_steps))
    print("  Saved to: {0}".format(output_path))


if __name__ == "__main__":
    main()
