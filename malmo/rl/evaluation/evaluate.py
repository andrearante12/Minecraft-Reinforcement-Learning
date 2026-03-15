"""
evaluation/evaluate.py
-----------------------
Load a trained checkpoint and evaluates the policy

Usage:
    cd parkour/
    python evaluation/evaluate.py --checkpoint checkpoints/ep1000.pt
    python evaluation/evaluate.py --checkpoint checkpoints/ep1000.pt --episodes 50
"""

import sys
import os
import argparse

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from training.config  import CFG
from models.mlp       import ActorCritic
from algos.ppo        import PPO
from Malmo.parkour.envs.simple_jump.parkour_env import ParkourEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--episodes", type=int, default=CFG.EVAL_EPISODES,
                        help="Number of evaluation episodes")
    return parser.parse_args()


def evaluate():
    args = parse_args()

    print("=" * 60)
    print("Parkour Policy Evaluation")
    print("=" * 60)
    print("Checkpoint:", args.checkpoint)
    print("Episodes:  ", args.episodes)
    print()

    env   = ParkourEnv(CFG)
    model = ActorCritic(CFG.INPUT_SIZE, CFG.HIDDEN_SIZE, CFG.N_ACTIONS)
    agent = PPO(model, CFG)
    agent.load(args.checkpoint)

    results = []

    for ep in range(1, args.episodes + 1):
        obs    = env.reset()
        done   = False
        total_reward = 0.0
        steps  = 0
        outcome = "timeout"

        while not done:
            # Greedy action — no exploration
            action = agent.select_action(obs, greedy=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps        += 1
            if done:
                outcome = info["outcome"]

        results.append((ep, total_reward, steps, outcome))
        print("  Ep {0:>3} | steps:{1:>3} | reward:{2:>7.2f} | {3}".format(
            ep, steps, total_reward, outcome))

    # ── Summary ───────────────────────────────────────────────────────────────
    n        = len(results)
    rewards  = [r[1] for r in results]
    steps    = [r[2] for r in results]
    outcomes = [r[3] for r in results]

    n_landed  = outcomes.count("landed")
    n_fell    = outcomes.count("fell")
    n_timeout = outcomes.count("timeout")

    print("\n" + "=" * 60)
    print("Evaluation Summary ({0} episodes)".format(n))
    print("=" * 60)
    print("  Success rate: {0:.1f}%  ({1}/{2})".format(100 * n_landed / n, n_landed, n))
    print("  Fell:         {0}".format(n_fell))
    print("  Timeout:      {0}".format(n_timeout))
    print("  Mean reward:  {0:.2f}".format(sum(rewards) / n))
    print("  Mean steps:   {0:.1f}".format(sum(steps) / n))
    print("  Min reward:   {0:.2f}".format(min(rewards)))
    print("  Max reward:   {0:.2f}".format(max(rewards)))

    env.close()


if __name__ == "__main__":
    evaluate()
