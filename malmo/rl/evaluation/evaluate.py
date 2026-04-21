"""
evaluation/evaluate.py
-----------------------
Load a trained checkpoint and evaluate the policy over TCP.

Requires env_server.py to be running first:
    Terminal 1: cd Malmo/Minecraft && launchClient.bat
    Terminal 2: conda activate malmo && python Malmo/rl/envs/env_server.py --env simple_jump --port 10002
    Terminal 3: conda activate train_env && python Malmo/rl/evaluation/evaluate.py --env simple_jump --checkpoint checkpoints/ppo_simple_jump_ep1000.pt --port 10002
"""

import sys
import os
import json
import argparse

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from training.configs.one_block_gap_cfg       import OneBlockGapCFG
from training.configs.simple_jump_cfg         import SimpleJumpCFG
from training.configs.three_block_gap_cfg     import ThreeBlockGapCFG
from training.configs.diagonal_small_cfg      import DiagonalSmallCFG
from training.configs.diagonal_medium_cfg     import DiagonalMediumCFG
from training.configs.vertical_small_cfg      import VerticalSmallCFG
from training.configs.multi_jump_course_cfg   import MultiJumpCourseCFG
from training.configs.multi_jump_branch_cfg   import MultiJumpBranchCFG

from models.actor_critic import ActorCritic
from algos.ppo           import PPO
from envs.env_client     import EnvClient

ENV_REGISTRY = {
    "one_block_gap":       OneBlockGapCFG,
    "simple_jump":         SimpleJumpCFG,
    "three_block_gap":     ThreeBlockGapCFG,
    "diagonal_small":      DiagonalSmallCFG,
    "diagonal_medium":     DiagonalMediumCFG,
    "vertical_small":      VerticalSmallCFG,
    "multi_jump_course":   MultiJumpCourseCFG,
    "multi_jump_branch":   MultiJumpBranchCFG,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="simple_jump",
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment to evaluate in (default: simple_jump)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--port", type=int, default=9999,
                        help="TCP port of the env server (default: 9999)")
    parser.add_argument("--save-successes", type=str, default=None,
                        metavar="PATH",
                        help="Save successful episodes as a replayable JSON (e.g. demos/eval_successes.json)")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    cfg  = ENV_REGISTRY[args.env]

    print("=" * 60)
    print("Parkour Policy Evaluation")
    print("=" * 60)
    print("Environment:", args.env)
    print("Checkpoint: ", args.checkpoint)
    print("Episodes:   ", args.episodes)
    print("Port:       ", args.port)
    print()

    env   = EnvClient(cfg.INPUT_SIZE, port=args.port)
    model = ActorCritic(cfg)
    agent = PPO(model, cfg)
    agent.load(args.checkpoint)

    results          = []
    success_episodes = []

    for ep in range(1, args.episodes + 1):
        obs    = env.reset()
        done   = False
        total_reward   = 0.0
        n_steps        = 0
        outcome        = "timeout"
        episode_actions = []

        while not done:
            action = agent.select_action(obs, greedy=True)
            episode_actions.append(int(action))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            n_steps      += 1
            if done:
                outcome = info["outcome"]

        jumps = info.get("jumps_completed", 0)
        results.append((ep, total_reward, n_steps, outcome, jumps))
        print("  Ep {0:>3} | steps:{1:>3} | reward:{2:>7.2f} | jumps:{3} | {4}".format(
            ep, n_steps, total_reward, jumps, outcome))

        if outcome == "landed" and args.save_successes:
            success_episodes.append({
                "outcome": outcome,
                "steps":   [{"action": a} for a in episode_actions],
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n        = len(results)
    rewards  = [r[1] for r in results]
    steps    = [r[2] for r in results]
    outcomes = [r[3] for r in results]
    jumps    = [r[4] for r in results]

    n_landed  = outcomes.count("landed")
    n_fell    = outcomes.count("fell")
    n_timeout = outcomes.count("timeout")
    n_near    = outcomes.count("near_miss")

    print("\n" + "=" * 60)
    print("Evaluation Summary ({0} episodes)".format(n))
    print("=" * 60)
    print("  Success rate: {0:.1f}%  ({1}/{2})".format(100 * n_landed / n, n_landed, n))
    print("  Fell:         {0}".format(n_fell))
    print("  Timeout:      {0}".format(n_timeout))
    print("  Near miss:    {0}".format(n_near))
    print("  Mean reward:  {0:.2f}".format(sum(rewards) / n))
    print("  Mean steps:   {0:.1f}".format(sum(steps) / n))
    print("  Min reward:   {0:.2f}".format(min(rewards)))
    print("  Max reward:   {0:.2f}".format(max(rewards)))

    max_jumps = max(jumps) if jumps else 0
    if max_jumps > 0:
        print("\n  Jumps completed:")
        for j in range(max_jumps + 1):
            count = jumps.count(j)
            print("    {0}: {1:>3}  ({2:.1f}%)".format(j, count, 100 * count / n))

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
