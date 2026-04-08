"""
training/train_sb3.py
---------------------
Train bridging (or other) environments using Stable Baselines3 PPO.

This is a separate training script from train.py — it uses SB3's native
training loop instead of our custom PPO implementation. The env server
setup is identical: launch Minecraft + env_server.py, then run this script.

Usage:
    # PPO from scratch:
    python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002

    # BC pre-training then PPO:
    python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --demo-path demos/bridging.json

    # Resume from checkpoint:
    python Malmo/rl/training/train_sb3.py --env bridging --base-port 10002 --checkpoint checkpoints/sb3_bridging_100000_steps.zip
"""

import sys
import os
import argparse
import json
import time
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList,
)

from envs.sb3_env_wrapper import MalmoGymEnv
from utils.logger import Logger

# ── Environment registry ─────────────────────────────────────────────────────
from training.configs.bridging_cfg import BridgingCFG

ENV_CONFIGS = {
    "bridging": BridgingCFG,
}


# ── Custom callbacks ─────────────────────────────────────────────────────────

class EpisodeLoggerCallback(BaseCallback):
    """Logs episode outcomes to our CSV logger for consistency with train.py."""

    def __init__(self, logger_obj, env_name, print_every=10, verbose=0):
        super().__init__(verbose)
        self.logger_obj = logger_obj
        self.env_name = env_name
        self.print_every = print_every
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done:
                self.episode_count += 1
                reward = info.get("episode", {}).get("r", 0.0)
                steps = info.get("episode", {}).get("l", 0)
                outcome = info.get("outcome", "unknown")

                # SB3 Monitor wrapper provides episode stats in info["episode"]
                # but our env provides outcome directly
                if "episode" not in info:
                    reward = 0.0
                    steps = 0

                self.logger_obj.log_episode(
                    self.episode_count, reward, steps,
                    outcome, self.env_name,
                )
                self.logger_obj.print_summary(every=self.print_every)
        return True


class TrajectoryLoggerCallback(BaseCallback):
    """Logs per-step trajectory data to _trajectories.csv for the visualizer."""

    def __init__(self, logger_obj, env_name, cfg, n_envs=1, verbose=0):
        super().__init__(verbose)
        self.logger_obj  = logger_obj
        self.env_name    = env_name
        self.cfg         = cfg
        self.n_envs      = n_envs
        self._ep_nums    = [0] * n_envs
        self._step_nums  = [0] * n_envs
        self._ep_counter = 0

    def _on_training_start(self):
        self.logger_obj.init_trajectory(self.env_name, self.cfg)

    def _on_step(self):
        infos   = self.locals.get("infos",   [])
        dones   = self.locals.get("dones",   [])
        rewards = self.locals.get("rewards", [])

        for i, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
            self._step_nums[i] += 1
            self.logger_obj.log_step(
                episode  = self._ep_nums[i],
                step     = self._step_nums[i],
                info     = info,
                reward   = float(reward),
                done     = bool(done),
                env_name = self.env_name,
            )
            if done:
                self._ep_counter   += 1
                self._ep_nums[i]    = self._ep_counter
                self._step_nums[i]  = 0
        return True


class InterruptSaveCallback(BaseCallback):
    """Saves model on KeyboardInterrupt."""

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self):
        return True

    def _on_training_end(self):
        pass


# ── BC pre-training ──────────────────────────────────────────────────────────

def bc_pretrain(model, demo_path, cfg, epochs=None, batch_size=None):
    """Pre-train SB3 PPO policy via behavioral cloning on demo data.

    Trains only the policy network (actor) using cross-entropy loss on
    recorded (obs, action) pairs. The value network is left untrained.
    """
    epochs = epochs or cfg.BC_EPOCHS
    batch_size = batch_size or cfg.BC_BATCH_SIZE
    device = model.policy.device

    # Load demos
    with open(demo_path, "r") as f:
        data = json.load(f)

    all_obs = []
    all_actions = []
    for episode in data["episodes"]:
        for step in episode["steps"]:
            all_obs.append(step["obs"])
            all_actions.append(step["action"])

    obs_np = np.array(all_obs, dtype=np.float32)
    act_np = np.array(all_actions, dtype=np.int64)
    n_demos = len(obs_np)
    print("BC pre-training: {0} transitions, {1} epochs".format(n_demos, epochs))

    # Use the policy's optimizer for the actor network
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.LR, eps=1e-5)

    for epoch in range(epochs):
        indices = np.random.permutation(n_demos)
        total_loss = 0.0
        total_correct = 0
        n_batches = 0

        for start in range(0, n_demos, batch_size):
            idx = indices[start:start + batch_size]
            obs_t = torch.tensor(obs_np[idx], dtype=torch.float32).to(device)
            act_t = torch.tensor(act_np[idx], dtype=torch.int64).to(device)

            # Get action logits from SB3 policy
            # SB3's MlpPolicy forward pass: features → action_net → logits
            features = policy.extract_features(obs_t, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)

            loss = nn.functional.cross_entropy(logits, act_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == act_t).sum().item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = total_correct / n_demos
        print("  BC epoch {0}/{1}: loss={2:.4f}  accuracy={3:.1%}".format(
            epoch + 1, epochs, avg_loss, accuracy))

    print("BC pre-training complete.")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO training")
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_CONFIGS.keys()),
                        help="Environment name")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--base-port", type=int, default=10002,
                        help="Base TCP port for env servers")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SB3 checkpoint (.zip) to resume from")
    parser.add_argument("--demo-path", type=str, default=None,
                        help="Path to demo JSON for BC pre-training")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total training timesteps (default: TOTAL_EPISODES * MAX_STEPS)")
    parser.add_argument("--bc-epochs", type=int, default=None,
                        help="BC pre-training epochs (default: cfg.BC_EPOCHS)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]

    n_envs = args.num_envs
    base_port = args.base_port
    total_timesteps = args.total_timesteps or (cfg.TOTAL_EPISODES * cfg.MAX_STEPS)

    # ── Create vectorized environment ─────────────────────────────────────
    def make_env(port):
        def _init():
            env = MalmoGymEnv(cfg, port)
            return Monitor(env)
        return _init

    env_fns = [make_env(base_port + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # ── Create or load model ──────────────────────────────────────────────
    checkpoint_dir = os.path.join(PARKOUR_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tb_log_dir = os.path.join(PARKOUR_ROOT, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    if args.checkpoint:
        print("Loading checkpoint: {0}".format(args.checkpoint))
        model = PPO.load(args.checkpoint, env=vec_env,
                         tensorboard_log=tb_log_dir)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.N_STEPS // n_envs,
            batch_size=cfg.BATCH_SIZE,
            n_epochs=cfg.N_EPOCHS,
            learning_rate=cfg.LR,
            clip_range=cfg.CLIP_EPS,
            ent_coef=cfg.ENTROPY_COEF,
            vf_coef=cfg.VALUE_COEF,
            gamma=cfg.GAMMA,
            gae_lambda=cfg.GAE_LAMBDA,
            max_grad_norm=cfg.MAX_GRAD_NORM,
            verbose=1,
            tensorboard_log=tb_log_dir,
        )

    # ── BC pre-training (optional) ────────────────────────────────────────
    if args.demo_path:
        bc_pretrain(model, args.demo_path, cfg, epochs=args.bc_epochs)

    # ── Callbacks ─────────────────────────────────────────────────────────
    run_name = "sb3_ppo_{0}".format(args.env)
    logger = Logger(
        log_dir=os.path.join(PARKOUR_ROOT, "logs"),
        run_name=run_name,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg.SAVE_EVERY * cfg.MAX_STEPS // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="sb3_{0}".format(args.env),
    )

    episode_callback = EpisodeLoggerCallback(
        logger_obj=logger,
        env_name=args.env,
        print_every=cfg.LOG_EVERY,
    )

    trajectory_callback = TrajectoryLoggerCallback(
        logger_obj = logger,
        env_name   = args.env,
        cfg        = cfg,
        n_envs     = n_envs,
    )

    callbacks = CallbackList([checkpoint_callback, episode_callback, trajectory_callback])

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SB3 PPO Training")
    print("=" * 60)
    print("Environment:      {0}".format(args.env))
    print("Num envs:         {0}".format(n_envs))
    print("Total timesteps:  {0}".format(total_timesteps))
    print("N_STEPS (per env):{0}".format(cfg.N_STEPS // n_envs))
    print("Batch size:       {0}".format(cfg.BATCH_SIZE))
    print("LR:               {0}".format(cfg.LR))
    print("Checkpoint:       {0}".format(args.checkpoint or "None"))
    print("Demo path:        {0}".format(args.demo_path or "None"))
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=args.checkpoint is None,
            tb_log_name="ppo_{0}".format(args.env),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_path = os.path.join(checkpoint_dir,
                                  "sb3_{0}_interrupted".format(args.env))
        model.save(save_path)
        print("Saved to: {0}.zip".format(save_path))
    finally:
        logger.close()
        vec_env.close()

    # Final save
    final_path = os.path.join(checkpoint_dir,
                               "sb3_{0}_final".format(args.env))
    model.save(final_path)
    print("Training complete. Final model: {0}.zip".format(final_path))


if __name__ == "__main__":
    main()
