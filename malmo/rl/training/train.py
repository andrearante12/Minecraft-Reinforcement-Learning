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
from models.mlp       import ActorCritic
from envs.env_client  import EnvClient
from utils.logger     import Logger
from training.curriculum import CurriculumScheduler

# ── Algorithm registry ────────────────────────────────────────────────────────
from algos.ppo import PPO
from algos.dqn import DQN

ALGO_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
}

# ── Environment registry ──────────────────────────────────────────────────────
# Tuples of (EnvClass, CfgClass) — EnvClass is not used directly by the client,
# but the scheduler needs the full registry for validation.
from training.configs.simple_jump_cfg      import SimpleJumpCFG
from training.configs.three_block_gap_cfg  import ThreeBlockGapCFG

ENV_REGISTRY = {
    "one_block_gap":   (None, OneBlockGapCFG),
    "simple_jump":     (None, SimpleJumpCFG),
    "three_block_gap": (None, ThreeBlockGapCFG),
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
    logger = Logger(cfg.LOG_DIR, run_name)

    env_label = args.curriculum if args.curriculum else args.env
    print_header(env_label, args.algo, cfg, num_envs=n_envs, total_episodes=total_episodes)

    start_episode = 1
    if args.checkpoint:
        agent.load(args.checkpoint)
        try:
            start_episode = int(args.checkpoint.split("ep")[-1].split(".")[0]) + 1
        except Exception:
            pass
        print("Resuming from episode {0}".format(start_episode))
        print()

    # Determine initial env for each slot
    initial_env = scheduler.env_for_episode(start_episode)
    current_envs = [initial_env] * n_envs

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
        while episode <= total_episodes:

            next_obs_all, rewards, dones, infos = agent.collect_steps(envs, obs_all)

            for i in range(n_envs):
                ep_rewards[i] += rewards[i]
                ep_steps[i]   += 1

                if dones[i]:
                    ep_outcome[i] = infos[i]["outcome"]

                if dones[i]:
                    logger.log_episode(episode, ep_rewards[i], int(ep_steps[i]),
                                       ep_outcome[i], env_name=current_envs[i])
                    logger.print_summary(every=cfg.LOG_EVERY)
                    print("  Ep {0:>4} | env:{1} | steps:{2:>3} | reward:{3:>7.2f} | outcome:{4}".format(
                        episode, current_envs[i], int(ep_steps[i]), ep_rewards[i], ep_outcome[i]))

                    # Switch env if curriculum says so
                    next_env = scheduler.env_for_episode(episode + 1)
                    if next_env != current_envs[i]:
                        envs[i].switch_env(next_env)
                        current_envs[i] = next_env

                    # Reset this env
                    next_obs_all[i] = envs[i].reset()
                    ep_rewards[i]   = 0.0
                    ep_steps[i]     = 0
                    ep_outcome[i]   = "timeout"

                    if episode % cfg.SAVE_EVERY == 0:
                        path = os.path.join(cfg.CHECKPOINT_DIR,
                                            "{0}_{1}_ep{2}.pt".format(args.algo, run_name, episode))
                        agent.save(path)

                    episode += 1
                    if episode > total_episodes:
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
                            "{0}_{1}_ep{2}_interrupted.pt".format(args.algo, run_name, episode))
        agent.save(path)

    finally:
        for env in envs:
            env.close()
        logger.close()
        print("Done.")


if __name__ == "__main__":
    train()
