"""
evaluation/evaluate_bridging.py
--------------------------------
Load a trained SB3 bridging checkpoint and evaluate the policy over TCP.

Requires env_server.py to be running first:
    Terminal 1: cd Malmo/Minecraft && launchClient.bat
    Terminal 2: conda activate malmo && python Malmo/rl/envs/env_server.py --env bridging --port 10002
    Terminal 3: conda activate train_env && python Malmo/rl/evaluation/evaluate_bridging.py --env bridging --checkpoint Malmo/rl/baselines/sb3_bridging_baseline.zip --episodes 10 --port 10002
"""

import sys
import os
import json
import argparse

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from stable_baselines3 import PPO
from envs.sb3_env_wrapper import MalmoGymEnv
from training.configs.bridging_cfg          import BridgingCFG
from training.configs.bridging_straight_cfg import BridgingStraightCFG
from training.configs.bridging_1block_cfg   import Bridging1BlockCFG
from training.configs.bridging_2block_cfg   import Bridging2BlockCFG
from training.configs.bridging_3block_cfg   import Bridging3BlockCFG
from training.configs.bridging_4block_cfg   import Bridging4BlockCFG

ENV_REGISTRY = {
    "bridging":          BridgingCFG,           # diagonal (GOAL_X_OFFSETS=[4])
    "bridging_straight": BridgingStraightCFG,   # straight (GOAL_X_OFFSETS=[0])
    "bridging_1block":   Bridging1BlockCFG,
    "bridging_2block":   Bridging2BlockCFG,
    "bridging_3block":   Bridging3BlockCFG,
    "bridging_4block":   Bridging4BlockCFG,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SB3 bridging checkpoint"
    )
    parser.add_argument("--env", type=str, default="bridging",
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment to evaluate in (default: bridging)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SB3 checkpoint (.zip file)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--port", type=int, default=10002,
                        help="TCP port of the env server (default: 10002)")
    parser.add_argument("--save-successes", type=str, default=None,
                        metavar="PATH",
                        help="Save successful episodes as replayable JSON")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    cfg  = ENV_REGISTRY[args.env]

    print("=" * 60)
    print("Bridging Policy Evaluation")
    print("=" * 60)
    print("Environment:", args.env)
    print("Checkpoint: ", args.checkpoint)
    print("Episodes:   ", args.episodes)
    print("Port:       ", args.port)
    print()

    model = PPO.load(args.checkpoint)
    env   = MalmoGymEnv(cfg, port=args.port)

    results          = []
    success_episodes = []

    for ep in range(1, args.episodes + 1):
        obs, _  = env.reset()
        done    = False
        total_reward    = 0.0
        n_steps         = 0
        outcome         = "timeout"
        episode_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            n_steps      += 1
            done = terminated or truncated
            if done:
                outcome = info.get("outcome", "unknown")

        blocks = info.get("blocks_placed", 0)
        results.append((ep, total_reward, n_steps, outcome, blocks))
        print("  Ep {0:>3} | steps:{1:>3} | reward:{2:>7.2f} | blocks:{3:>3} | {4}".format(
            ep, n_steps, total_reward, blocks, outcome))

        if outcome == "landed" and args.save_successes:
            success_episodes.append({
                "outcome": outcome,
                "steps":   [{"action": a} for a in episode_actions],
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n        = len(results)
    rewards  = [r[1] for r in results]
    steps_   = [r[2] for r in results]
    outcomes = [r[3] for r in results]
    blocks_  = [r[4] for r in results]

    n_landed  = outcomes.count("landed")
    n_fell    = outcomes.count("fell")
    n_timeout = outcomes.count("timeout")
    n_near    = outcomes.count("near_miss")

    print("\n" + "=" * 60)
    print("Evaluation Summary ({0} episodes)".format(n))
    print("=" * 60)
    print("  Success rate:  {0:.1f}%  ({1}/{2})".format(100 * n_landed / n, n_landed, n))
    print("  Fell:          {0}".format(n_fell))
    print("  Timeout:       {0}".format(n_timeout))
    print("  Near miss:     {0}".format(n_near))
    print("  Mean reward:   {0:.2f}".format(sum(rewards) / n))
    print("  Mean steps:    {0:.1f}".format(sum(steps_) / n))
    print("  Mean blocks:   {0:.1f}".format(sum(blocks_) / n))
    print("  Min reward:    {0:.2f}".format(min(rewards)))
    print("  Max reward:    {0:.2f}".format(max(rewards)))

    if args.save_successes and success_episodes:
        save_path = args.save_successes
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"env": args.env, "episodes": success_episodes}, f)
        print("\n  Saved {0} successful episode(s) to {1}".format(
            len(success_episodes), save_path))
    elif args.save_successes:
        print("\n  No successful episodes to save.")

    env.close()


if __name__ == "__main__":
    evaluate()
