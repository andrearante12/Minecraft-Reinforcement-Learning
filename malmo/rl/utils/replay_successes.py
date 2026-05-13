"""
utils/replay_successes.py
-------------------------
Replay successful episodes from a training trajectories CSV.

Training saves every step (including actions) to *_trajectories.csv via
TrajectoryLoggerCallback. This script reads that file, finds episodes that
ended with outcome='landed', and replays their action sequences through a
running env server so you can watch them in the Minecraft client.

Usage:
    conda activate train_env

    # Replay all successful episodes:
    python Malmo/rl/utils/replay_successes.py --trajectories Malmo/rl/logs/<run>_trajectories.csv --port 10002

    # Replay a specific episode:
    python Malmo/rl/utils/replay_successes.py --trajectories Malmo/rl/logs/<run>_trajectories.csv --port 10002 --episode 42

    # Slow down playback (0.5 = half speed):
    python Malmo/rl/utils/replay_successes.py --trajectories Malmo/rl/logs/<run>_trajectories.csv --port 10002 --speed 0.5
"""

import sys
import os
import csv
import argparse
import time
import functools

print = functools.partial(print, flush=True)

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from envs.env_client import EnvClient
from training.configs.bridging_cfg import BridgingCFG


def load_successful_episodes(csv_path):
    """
    Parse a trajectories CSV and return action sequences for successful episodes.

    Returns:
        successes: dict of {episode_num: [action_name, ...]}
        all_outcomes: dict of {episode_num: outcome_string}
    """
    episodes  = {}   # episode -> list of (step, action_name)
    outcomes  = {}   # episode -> outcome (last seen for that episode)

    rewards = {}   # episode -> cumulative reward

    with open(csv_path, newline='', encoding='utf-8') as f:
        # Skip comment lines at the top (#) before handing to DictReader
        non_comment = (row for row in f if not row.startswith('#'))
        reader = csv.DictReader(non_comment)
        for row in reader:
            ep      = int(row['episode'])
            step    = int(row['step'])
            action  = row['action']
            outcome = row['outcome']

            if ep not in episodes:
                episodes[ep] = []
                rewards[ep]  = 0.0
            episodes[ep].append((step, action))
            rewards[ep] += float(row['reward'])

            # outcome is non-empty only on the terminal step
            if outcome:
                outcomes[ep] = outcome

    successes = {
        ep: [a for _, a in sorted(steps, key=lambda x: x[0])]
        for ep, steps in episodes.items()
        if outcomes.get(ep) == 'landed'
    }
    return successes, outcomes, rewards


def build_action_map(cfg):
    """Map action name -> action index from cfg.ACTIONS."""
    return {name: i for i, (name, _, _) in enumerate(cfg.ACTIONS)}


def replay_episode(client, actions, action_map, step_duration):
    obs = client.reset()
    print("  reset done, replaying {0} steps...".format(len(actions)))

    for action_name in actions:
        action_idx = action_map.get(action_name)
        if action_idx is None:
            print("  WARNING: unknown action '{0}', skipping".format(action_name))
            continue
        obs, reward, done, info = client.step(action_idx)
        time.sleep(step_duration)
        if done:
            print("  episode ended: {0}".format(info.get('outcome', '?')))
            break


def main():
    parser = argparse.ArgumentParser(description="Replay successful episodes from a trajectories CSV")
    parser.add_argument('--trajectories', required=True,
                        help='Path to *_trajectories.csv from a training run')
    parser.add_argument('--port', type=int, default=10002,
                        help='TCP port of the running env server')
    parser.add_argument('--episode', type=int, default=None,
                        help='Replay a specific episode number (default: all successes)')
    parser.add_argument('--top', type=int, default=None,
                        help='Only replay the N highest-reward successful episodes')
    parser.add_argument('--min-reward', type=float, default=None,
                        help='Only replay episodes with total reward >= this value')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (0.5 = half speed, 2.0 = double speed)')
    args = parser.parse_args()

    cfg = BridgingCFG
    step_duration = cfg.STEP_DURATION / max(args.speed, 1e-6)

    print("Loading trajectories from: {0}".format(args.trajectories))
    successes, all_outcomes, rewards = load_successful_episodes(args.trajectories)

    if not successes:
        print("No successful (outcome='landed') episodes found.")
        return

    print("Found {0} successful episodes out of {1} total.".format(
        len(successes), len(all_outcomes)))

    if args.episode is not None:
        if args.episode not in successes:
            print("Episode {0} not found or not successful.".format(args.episode))
            print("Successful episodes: {0}".format(sorted(successes.keys())))
            return
        to_replay = {args.episode: successes[args.episode]}
    else:
        to_replay = dict(successes)

        # Filter by minimum reward
        if args.min_reward is not None:
            to_replay = {ep: acts for ep, acts in to_replay.items()
                         if rewards.get(ep, 0.0) >= args.min_reward}
            print("After --min-reward {0}: {1} episodes".format(args.min_reward, len(to_replay)))

        # Keep only top-N by reward
        if args.top is not None:
            ranked = sorted(to_replay.keys(), key=lambda ep: rewards.get(ep, 0.0), reverse=True)
            to_replay = {ep: to_replay[ep] for ep in ranked[:args.top]}
            print("Top {0} by reward:".format(args.top))
            for ep in sorted(to_replay, key=lambda ep: rewards.get(ep, 0.0), reverse=True):
                print("  ep {0:>5}  reward={1:.1f}".format(ep, rewards.get(ep, 0.0)))

    action_map = build_action_map(cfg)

    print("Connecting to env server at 127.0.0.1:{0}...".format(args.port))
    client = EnvClient(cfg.INPUT_SIZE, port=args.port)

    try:
        for ep_num, actions in sorted(to_replay.items(), key=lambda x: rewards.get(x[0], 0.0), reverse=True):
            print("\nEpisode {0}  reward={1:.1f}  steps={2}".format(
                ep_num, rewards.get(ep_num, 0.0), len(actions)))
            replay_episode(client, actions, action_map, step_duration)
            time.sleep(1.5)   # pause between episodes so the reset is visible
    finally:
        client.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
