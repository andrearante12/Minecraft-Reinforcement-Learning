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

from training.configs.one_block_gap_cfg import OneBlockGapCFG


PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from models.mlp       import ActorCritic
from envs.env_client  import EnvClient
from utils.logger     import Logger

# ── Algorithm registry ────────────────────────────────────────────────────────
from algos.ppo import PPO
from algos.dqn import DQN

ALGO_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
}

# ── Environment registry ──────────────────────────────────────────────────────
# Only configs are needed here — the client is generic
from training.configs.simple_jump_cfg      import SimpleJumpCFG
from training.configs.three_block_gap_cfg  import ThreeBlockGapCFG

ENV_REGISTRY = {
    "one_block_gap":   OneBlockGapCFG,
    "simple_jump":     SimpleJumpCFG,
    "three_block_gap": ThreeBlockGapCFG,
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
    return parser.parse_args()


def print_header(env_name, algo_name, cfg, num_envs=1):
    print("=" * 60)
    print("Malmo RL Training")
    print("=" * 60)
    print("Environment:    ", env_name)
    print("Algorithm:      ", algo_name.upper())
    print("Device:         ", "cuda" if torch.cuda.is_available() else "cpu")
    print("Num envs:       ", num_envs)
    print("Total episodes: ", cfg.TOTAL_EPISODES)
    print("Obs size:       ", cfg.INPUT_SIZE)
    print("N actions:      ", cfg.N_ACTIONS)
    print("Hidden size:    ", cfg.HIDDEN_SIZE)
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
    print("=" * 60)
    print()


def print_update(losses):
    metrics = " | ".join("{0}:{1:.4f}".format(k, v) for k, v in losses.items())
    print("  [update] {0}".format(metrics))


def train():
    args    = parse_args()
    cfg     = ENV_REGISTRY[args.env]
    n_envs  = args.num_envs

    # Validate N_STEPS divisibility for PPO
    if args.algo == "ppo" and cfg.N_STEPS % n_envs != 0:
        print("ERROR: N_STEPS ({0}) must be divisible by --num-envs ({1})".format(
            cfg.N_STEPS, n_envs))
        sys.exit(1)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Create N env clients, each connecting to base_port+i
    base_port = args.base_port
    envs   = [EnvClient(cfg.INPUT_SIZE, port=base_port + i) for i in range(n_envs)]
    model  = ActorCritic(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE, cfg.N_ACTIONS)
    agent  = ALGO_REGISTRY[args.algo](model, cfg, n_envs=n_envs)
    logger = Logger(cfg.LOG_DIR, "{0}_{1}".format(args.env, args.algo))

    print_header(args.env, args.algo, cfg, num_envs=n_envs)

    start_episode = 1
    if args.checkpoint:
        agent.load(args.checkpoint)
        try:
            start_episode = int(args.checkpoint.split("ep")[-1].split(".")[0]) + 1
        except Exception:
            pass
        print("Resuming from episode {0}".format(start_episode))
        print()

    # Per-env state tracking
    obs_all    = np.zeros((n_envs, cfg.INPUT_SIZE), dtype=np.float32)
    ep_rewards = np.zeros(n_envs, dtype=np.float64)
    ep_steps   = np.zeros(n_envs, dtype=np.int64)
    ep_outcome = ["timeout"] * n_envs

    for i, env in enumerate(envs):
        obs_all[i] = env.reset()

    episode      = start_episode
    update_count = 0

    print("Starting training... (Ctrl+C to stop and save)")
    print()

    try:
        while episode <= cfg.TOTAL_EPISODES:

            next_obs_all, rewards, dones, infos = agent.collect_steps(envs, obs_all)

            for i in range(n_envs):
                ep_rewards[i] += rewards[i]
                ep_steps[i]   += 1

                if dones[i]:
                    ep_outcome[i] = infos[i]["outcome"]

                if dones[i]:
                    logger.log_episode(episode, ep_rewards[i], int(ep_steps[i]), ep_outcome[i])
                    logger.print_summary(every=cfg.LOG_EVERY)
                    print("  Ep {0:>4} | steps:{1:>3} | reward:{2:>7.2f} | outcome:{3}".format(
                        episode, int(ep_steps[i]), ep_rewards[i], ep_outcome[i]))

                    # Reset this env
                    next_obs_all[i] = envs[i].reset()
                    ep_rewards[i]   = 0.0
                    ep_steps[i]     = 0
                    ep_outcome[i]   = "timeout"

                    if episode % cfg.SAVE_EVERY == 0:
                        path = os.path.join(cfg.CHECKPOINT_DIR,
                                            "{0}_{1}_ep{2}.pt".format(args.algo, args.env, episode))
                        agent.save(path)

                    episode += 1
                    if episode > cfg.TOTAL_EPISODES:
                        break

            obs_all = next_obs_all

            if agent.buffer_full():
                losses = agent.update(last_obs=obs_all)
                update_count += 1
                logger.log_update(**losses)
                if update_count % 10 == 0:
                    print_update(losses)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        path = os.path.join(cfg.CHECKPOINT_DIR,
                            "{0}_{1}_ep{2}_interrupted.pt".format(args.algo, args.env, episode))
        agent.save(path)

    finally:
        for env in envs:
            env.close()
        logger.close()
        print("Done.")


if __name__ == "__main__":
    train()
