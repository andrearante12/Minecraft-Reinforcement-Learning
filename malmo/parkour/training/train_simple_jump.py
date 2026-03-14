"""
training/train_simple_jump.py
------------------------------
Training entrypoint for the simple jump task.

Usage:
    python parkour/training/train_simple_jump.py
    python parkour/training/train_simple_jump.py --checkpoint checkpoints/ep500.pt
"""

import sys
import os
import argparse

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from training.config  import CFG
from models.mlp       import ActorCritic
from algos.ppo        import PPO
from envs.env_client  import ParkourEnvClient
from utils.logger     import Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def train():
    args = parse_args()

    print("=" * 60)
    print("Parkour PPO Training — Simple Jump")
    print("=" * 60)
    print("Device:       ", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    print("Total episodes:", CFG.TOTAL_EPISODES)
    print("N steps/update:", CFG.N_STEPS)
    print("Obs size:      ", CFG.INPUT_SIZE)
    print("Actions:       ", CFG.N_ACTIONS)
    print()

    # Init
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.LOG_DIR, exist_ok=True)

    env    = ParkourEnvClient()          # connects to env_server.py over socket
    model  = ActorCritic(CFG.INPUT_SIZE, CFG.HIDDEN_SIZE, CFG.N_ACTIONS)
    agent  = PPO(model, CFG)
    logger = Logger(CFG.LOG_DIR, "simple_jump_ppo")

    # Resume from checkpoint if provided
    start_episode = 1
    if args.checkpoint:
        agent.load(args.checkpoint)
        try:
            start_episode = int(args.checkpoint.split("ep")[-1].split(".")[0]) + 1
            print("Resuming from episode", start_episode)
        except Exception:
            pass

    # Training Loop
    obs = env.reset()

    episode    = start_episode
    ep_reward  = 0.0
    ep_steps   = 0
    ep_outcome = "timeout"

    print("Starting training... (Ctrl+C to stop and save)")
    print()

    try:
        while episode <= CFG.TOTAL_EPISODES:

            # One step
            next_obs, reward, done, info = agent.collect_step(env, obs)
            ep_reward += reward
            ep_steps  += 1
            obs        = next_obs

            if done:
                ep_outcome = info["outcome"]

            # episode reaches terminal state
            if done:
                logger.log_episode(episode, ep_reward, ep_steps, ep_outcome)
                logger.print_summary(every=CFG.LOG_EVERY)

                print("  Ep {0:>4} | steps:{1:>3} | reward:{2:>7.2f} | {3}".format(
                    episode, ep_steps, ep_reward, ep_outcome))

                obs        = env.reset()
                ep_reward  = 0.0
                ep_steps   = 0
                ep_outcome = "timeout"
                episode   += 1

            # PPO update when buffer is full
            if agent.buffer_full():
                losses = agent.update(last_obs=obs)
                logger.log_update(
                    losses["policy_loss"],
                    losses["value_loss"],
                    losses["entropy"],
                )

            # Periodic checkpoint 
            if episode % CFG.SAVE_EVERY == 0 and done:
                path = os.path.join(CFG.CHECKPOINT_DIR,
                                    "ep{0}.pt".format(episode))
                agent.save(path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        path = os.path.join(CFG.CHECKPOINT_DIR,
                            "ep{0}_interrupted.pt".format(episode))
        agent.save(path)

    finally:
        env.close()
        logger.close()
        print("Done.")


if __name__ == "__main__":
    train()