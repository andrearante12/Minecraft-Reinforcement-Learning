"""
algos/ppo.py
------------
Proximal Policy Optimization — implements BaseAgent.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algos.base_agent import BaseAgent


class RolloutBuffer:
    def __init__(self, n_steps, obs_size):
        self.n_steps  = n_steps
        self.ptr      = 0
        self.full     = False
        self.obs       = np.zeros((n_steps, obs_size), dtype=np.float32)
        self.actions   = np.zeros(n_steps, dtype=np.int64)
        self.rewards   = np.zeros(n_steps, dtype=np.float32)
        self.dones     = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values    = np.zeros(n_steps, dtype=np.float32)

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def is_full(self):
        return self.full

    def clear(self):
        self.ptr  = 0
        self.full = False

    def get_tensors(self, device):
        return (
            torch.tensor(self.obs,       dtype=torch.float32).to(device),
            torch.tensor(self.actions,   dtype=torch.int64).to(device),
            torch.tensor(self.rewards,   dtype=torch.float32).to(device),
            torch.tensor(self.dones,     dtype=torch.float32).to(device),
            torch.tensor(self.log_probs, dtype=torch.float32).to(device),
            torch.tensor(self.values,    dtype=torch.float32).to(device),
        )


def compute_returns(rewards, dones, last_value, gamma):
    n       = len(rewards)
    returns = torch.zeros(n, dtype=torch.float32)
    R       = last_value
    for t in reversed(range(n)):
        R          = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R
    return returns


class PPO(BaseAgent):
    def __init__(self, model, cfg):
        self.model     = model
        self.cfg       = cfg
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)
        self.buffer    = RolloutBuffer(cfg.N_STEPS, cfg.INPUT_SIZE)
        print("PPO initialized on device:", self.device)

    def collect_step(self, env, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist     = self.model.get_distribution(obs_t)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            value    = self.model.get_value(obs_t)
        next_obs, reward, done, info = env.step(action.item())
        self.buffer.add(obs, action.item(), reward, done, log_prob.item(), value.item())
        return next_obs, reward, done, info

    def buffer_full(self):
        return self.buffer.is_full()

    def update(self, last_obs):
        with torch.no_grad():
            last_val = self.model.get_value(
                torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).item()

        obs_t, actions_t, rewards_t, dones_t, old_log_probs_t, values_t = \
            self.buffer.get_tensors(self.device)

        returns    = compute_returns(rewards_t.cpu(), dones_t.cpu(), last_val, self.cfg.GAMMA).to(self.device)
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(self.cfg.N_STEPS)
        total_policy_loss = total_value_loss = total_entropy = n_updates = 0

        for _ in range(self.cfg.N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, self.cfg.N_STEPS, self.cfg.BATCH_SIZE):
                idx = indices[start:start + self.cfg.BATCH_SIZE]

                new_log_probs, new_values, entropy = self.model.evaluate_actions(
                    obs_t[idx], actions_t[idx]
                )

                ratio       = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1       = ratio * advantages[idx]
                surr2       = torch.clamp(ratio, 1 - self.cfg.CLIP_EPS, 1 + self.cfg.CLIP_EPS) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(new_values, returns[idx])
                loss        = policy_loss + self.cfg.VALUE_COEF * value_loss - self.cfg.ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                n_updates         += 1

        self.buffer.clear()
        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss":  total_value_loss  / max(n_updates, 1),
            "entropy":     total_entropy     / max(n_updates, 1),
        }

    def select_action(self, obs, greedy=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if greedy:
                return self.model(obs_t)[0].argmax(dim=-1).item()
            return self.model.get_distribution(obs_t).sample().item()