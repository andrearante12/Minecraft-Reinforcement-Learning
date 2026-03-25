"""
algos/behavioral_cloning.py
---------------------------
Behavioral Cloning — supervised learning from human demonstrations.

Trains the ActorCritic policy head via cross-entropy loss on recorded
(observation, action) pairs. The critic head receives no gradient from BC;
it will be trained from scratch during PPO fine-tuning.

The BC -> PPO transition is seamless: load the BC checkpoint with
    --checkpoint bc_model.pt --algo ppo
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algos.base_agent import BaseAgent


class BehavioralCloning(BaseAgent):
    def __init__(self, model, cfg, n_envs=1):
        self.model   = model
        self.cfg     = cfg
        self.n_envs  = n_envs
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)

        # Load demonstration data
        if not cfg.DEMO_PATH:
            raise ValueError("BC requires --demo-path pointing to a demo JSON file")
        self.demo_obs, self.demo_actions = self._load_demos(cfg.DEMO_PATH)
        self.n_demos = len(self.demo_obs)

        # Step counter to trigger updates periodically
        self._steps_since_update = 0
        self._update_every = cfg.N_STEPS  # reuse PPO's N_STEPS as collection interval

        print("BehavioralCloning initialized on device:", self.device)
        print("  Demo samples: {0}".format(self.n_demos))
        print("  BC epochs per update: {0}".format(cfg.BC_EPOCHS))
        print("  BC batch size: {0}".format(cfg.BC_BATCH_SIZE))

    def _load_demos(self, path):
        """Load demo JSON and flatten all episodes into (obs, action) arrays."""
        with open(path, "r") as f:
            data = json.load(f)

        all_obs = []
        all_actions = []
        for episode in data["episodes"]:
            for step in episode["steps"]:
                all_obs.append(step["obs"])
                all_actions.append(step["action"])

        obs = np.array(all_obs, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.int64)
        print("Loaded {0} demo transitions from {1}".format(len(obs), path))
        return obs, actions

    def collect_step(self, env, obs):
        """Play current policy in env for evaluation/logging."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.model.get_distribution(obs_t)
            action = dist.sample().item()

        next_obs, reward, done, info = env.step(action)
        self._steps_since_update += 1
        return next_obs, reward, done, info

    def collect_steps(self, envs, obs_all):
        """Batched collection: forward pass for all envs, then step each."""
        n_envs = len(envs)

        obs_t = torch.tensor(obs_all, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.model.get_distribution(obs_t)
            actions = dist.sample()

        actions_np = actions.cpu().numpy()

        next_obs_all = np.zeros_like(obs_all)
        rewards = np.zeros(n_envs, dtype=np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        infos = [None] * n_envs

        for i, env in enumerate(envs):
            next_obs, reward, done, info = env.step(int(actions_np[i]))
            next_obs_all[i] = next_obs
            rewards[i] = reward
            dones[i] = float(done)
            infos[i] = info

        self._steps_since_update += 1
        return next_obs_all, rewards, dones, infos

    def buffer_full(self):
        """Trigger an update every N collection steps."""
        if self._steps_since_update >= self._update_every:
            self._steps_since_update = 0
            return True
        return False

    def update(self, last_obs):
        """Train policy on demo data via cross-entropy loss."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        indices = np.arange(self.n_demos)
        batch_size = self.cfg.BC_BATCH_SIZE

        for epoch in range(self.cfg.BC_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, self.n_demos, batch_size):
                idx = indices[start:start + batch_size]

                obs_batch = torch.tensor(self.demo_obs[idx],
                                         dtype=torch.float32).to(self.device)
                act_batch = torch.tensor(self.demo_actions[idx],
                                         dtype=torch.int64).to(self.device)

                logits, _ = self.model(obs_batch)
                loss = nn.functional.cross_entropy(logits, act_batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == act_batch).sum().item()
                total_samples += len(idx)

        n_batches = max(total_samples // batch_size, 1)
        return {
            "bc_loss":  total_loss / n_batches,
            "accuracy": total_correct / max(total_samples, 1),
        }

    def select_action(self, obs, greedy=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if greedy:
                logits, _ = self.model(obs_t)
                return logits.argmax(dim=-1).item()
            return self.model.get_distribution(obs_t).sample().item()
