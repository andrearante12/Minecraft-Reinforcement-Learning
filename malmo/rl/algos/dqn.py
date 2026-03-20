"""
algos/dqn.py
------------
Deep Q-Network — implements BaseAgent.

DQN is an off-policy algorithm: it stores transitions in a replay buffer
and samples random minibatches for updates, decoupling data collection
from learning. This is the key difference from PPO which is on-policy.

Key components:
  - ReplayBuffer: stores past transitions, sampled randomly during update
  - Q-network:    predicts Q(s,a) for all actions given a state
  - Target network: a frozen copy of the Q-network updated periodically
                    to stabilize training (reduces oscillation)
  - Epsilon-greedy: exploration strategy — take a random action with
                    probability epsilon, greedy action otherwise.
                    Epsilon decays over training as the agent improves.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from algos.base_agent import BaseAgent


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQN(BaseAgent):
    def __init__(self, model, cfg, n_envs=1):
        self.model     = model
        self.cfg       = cfg
        self.n_envs    = n_envs
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)

        # Target network — frozen copy, synced every TARGET_UPDATE_FREQ steps
        import copy
        self.target_model = copy.deepcopy(model)
        self.target_model.to(self.device)
        self.target_model.eval()

        self.buffer       = ReplayBuffer(cfg.BUFFER_CAPACITY)
        self.epsilon      = cfg.EPSILON_START
        self.steps_done   = 0

        print("DQN initialized on device:", self.device)
        print("Epsilon start: {0} | Epsilon end: {1} | Decay steps: {2}".format(
            cfg.EPSILON_START, cfg.EPSILON_END, cfg.EPSILON_DECAY_STEPS))

    def _decay_epsilon(self):
        """Linear epsilon decay."""
        progress = min(self.steps_done / self.cfg.EPSILON_DECAY_STEPS, 1.0)
        self.epsilon = self.cfg.EPSILON_START + progress * (
            self.cfg.EPSILON_END - self.cfg.EPSILON_START
        )

    def _sync_target(self):
        """Copy online network weights to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def collect_step(self, env, obs):
        """Epsilon-greedy action selection — no distribution needed."""
        self.steps_done += 1
        self._decay_epsilon()

        if random.random() < self.epsilon:
            action = random.randint(0, self.cfg.N_ACTIONS - 1)
        else:
            action = self.select_action(obs, greedy=True)

        next_obs, reward, done, info = env.step(action)
        self.buffer.add(obs, action, reward, next_obs, done)
        return next_obs, reward, done, info

    def collect_steps(self, envs, obs_all):
        """Collect one step from each env independently."""
        n_envs = len(envs)
        next_obs_all = np.zeros_like(obs_all)
        rewards = np.zeros(n_envs, dtype=np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        infos = [None] * n_envs
        for i, env in enumerate(envs):
            next_obs, reward, done, info = self.collect_step(env, obs_all[i])
            next_obs_all[i] = next_obs
            rewards[i] = reward
            dones[i] = float(done)
            infos[i] = info
        return next_obs_all, rewards, dones, infos

    def buffer_full(self):
        """DQN can update as soon as the buffer has enough samples."""
        return len(self.buffer) >= self.cfg.BATCH_SIZE

    def update(self, last_obs=None):
        """Sample a minibatch and update the Q-network."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.BATCH_SIZE)

        obs_t      = torch.tensor(obs,      dtype=torch.float32).to(self.device)
        actions_t  = torch.tensor(actions,  dtype=torch.int64).to(self.device)
        rewards_t  = torch.tensor(rewards,  dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones_t    = torch.tensor(dones,    dtype=torch.float32).to(self.device)

        # Current Q values for taken actions
        # model returns (logits, value) — use logits as Q-values for DQN
        q_values = self.model(obs_t)[0]
        q_taken  = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values — use frozen target network
        with torch.no_grad():
            next_q      = self.target_model(next_obs_t)[0].max(dim=1).values
            q_targets   = rewards_t + self.cfg.GAMMA * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_taken, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()

        # Periodically sync target network
        if self.steps_done % self.cfg.TARGET_UPDATE_FREQ == 0:
            self._sync_target()

        return {
            "q_loss":   loss.item(),
            "epsilon":  self.epsilon,
            "q_mean":   q_values.mean().item(),
        }

    def select_action(self, obs, greedy=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(obs_t)[0]
            return q_values.argmax(dim=-1).item()

    # ── Extra state for checkpointing ─────────────────────────────────────────
    def _extra_state(self):
        return {
            "epsilon":    self.epsilon,
            "steps_done": self.steps_done,
        }

    def _load_extra_state(self, state):
        self.epsilon    = state.get("epsilon",    self.cfg.EPSILON_START)
        self.steps_done = state.get("steps_done", 0)
