"""
training/train_simple_jump.py
------------------------------
Training entrypoint for the simple jump task.

Requires env_server.py to be running in the malmo env first:
    Terminal 1: conda activate malmo && python parkour/envs/env_server.py
    Terminal 2: conda activate train_env && python parkour/training/train_simple_jump.py

Usage:
    python parkour/training/train_simple_jump.py
    python parkour/training/train_simple_jump.py --algo dqn
    python parkour/training/train_simple_jump.py --checkpoint checkpoints/ep500.pt
"""

import sys
import os
import argparse
import torch

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from training.config  import CFG
from models.mlp       import ActorCritic
from envs.env_client  import ParkourEnvClient
from utils.logger     import Logger

from algos.ppo import PPO
from algos.dqn import DQN

ALGO_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       type=str, default="ppo",
                        choices=list(ALGO_REGISTRY.keys()),
                        help="RL algorithm to use (default: ppo)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def print_header(algo_name, cfg):
    print("=" * 60)
    print("Parkour RL Training — Simple Jump")
    print("=" * 60)
    print("Algorithm:      ", algo_name.upper())
    print("Device:         ", "cuda" if torch.cuda.is_available() else "cpu")
    print("Total episodes: ", cfg.TOTAL_EPISODES)
    print("Obs size:       ", cfg.INPUT_SIZE)
    print("N actions:      ", cfg.N_ACTIONS)
    print("Hidden size:    ", cfg.HIDDEN_SIZE)
    if algo_name == "ppo":
        print("N steps/update: ", cfg.N_STEPS)
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
    print("=" * 60)
    print()


def print_update(losses):
    """Print whatever keys the algorithm's update() returned."""
    metrics = " | ".join("{0}:{1:.4f}".format(k, v) for k, v in losses.items())
    print("  [update] {0}".format(metrics))


def train():
    args = parse_args()

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.LOG_DIR, exist_ok=True)

    env    = ParkourEnvClient()
    model  = ActorCritic(CFG.INPUT_SIZE, CFG.HIDDEN_SIZE, CFG.N_ACTIONS)
    agent  = ALGO_REGISTRY[args.algo](model, CFG)
    logger = Logger(CFG.LOG_DIR, "simple_jump_{0}".format(args.algo))

    print_header(args.algo, CFG)

    # Resume from checkpoint
    start_episode = 1
    if args.checkpoint:
        agent.load(args.checkpoint)
        try:
            start_episode = int(args.checkpoint.split("ep")[-1].split(".")[0]) + 1
        except Exception:
            pass
        print("Resuming from episode {0}".format(start_episode))
        print()

    # ── Training loop ─────────────────────────────────────────────────────────
    obs          = env.reset()
    episode      = start_episode
    ep_reward    = 0.0
    ep_steps     = 0
    ep_outcome   = "timeout"
    update_count = 0

    print("Starting training... (Ctrl+C to stop and save)")
    print()

    try:
        while episode <= CFG.TOTAL_EPISODES:

            # ── Collect one step ───────────────────────────────────────────
            next_obs, reward, done, info = agent.collect_step(env, obs)
            ep_reward += reward
            ep_steps  += 1
            obs        = next_obs

            if done:
                ep_outcome = info["outcome"]

            # ── Episode ended ──────────────────────────────────────────────
            if done:
                logger.log_episode(episode, ep_reward, ep_steps, ep_outcome)
                logger.print_summary(every=CFG.LOG_EVERY)

                print("  Ep {0:>4} | steps:{1:>3} | reward:{2:>7.2f} | outcome:{3}".format(
                    episode, ep_steps, ep_reward, ep_outcome))

                obs        = env.reset()
                ep_reward  = 0.0
                ep_steps   = 0
                ep_outcome = "timeout"
                episode   += 1

            # ── Update ────────────────────────────────────────────────────
            if agent.buffer_full():
                losses = agent.update(last_obs=obs)
                update_count += 1
                logger.log_update(**losses)
                if update_count % 10 == 0:
                    print_update(losses)

            # ── Periodic checkpoint ────────────────────────────────────────
            if episode % CFG.SAVE_EVERY == 0 and done:
                path = os.path.join(CFG.CHECKPOINT_DIR,
                                    "{0}_ep{1}.pt".format(args.algo, episode))
                agent.save(path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        path = os.path.join(CFG.CHECKPOINT_DIR,
                            "{0}_ep{1}_interrupted.pt".format(args.algo, episode))
        agent.save(path)

    finally:
        env.close()
        logger.close()
        print("Done.")


if __name__ == "__main__":
    train()