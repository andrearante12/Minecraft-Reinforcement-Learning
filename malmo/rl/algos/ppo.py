"""
algos/ppo.py
------------
Proximal Policy Optimization — on-policy, learns data from data collected by the most recent
version of the policy
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algos.base_agent import BaseAgent

# Temporary storage unit that holds enough data for a single update cycle, data is discarded afterwards
# Stores a collection of transitions (records of experience at a time t):
#   - obs: the state
#   - actions: action taken in that state
#   - rewards: immediate reward given by enviornment
#   - dones: 0 if non-terminal state, 1 if terminal state
#   - log_prob: confidence -> Log(P) taken under the old policy used for PPO ratio
#   - values: prediction of future rewards from this state (according to Critic)
class RolloutBuffer:
    def __init__(self, n_steps, obs_size, n_envs=1):
        self.n_envs        = n_envs
        self.n_steps_per_env = n_steps // n_envs
        self.total_steps   = self.n_steps_per_env * n_envs
        self.ptr           = 0
        self.full          = False
        self.obs       = np.zeros((self.n_steps_per_env, n_envs, obs_size), dtype=np.float32)
        self.actions   = np.zeros((self.n_steps_per_env, n_envs), dtype=np.int64)
        self.rewards   = np.zeros((self.n_steps_per_env, n_envs), dtype=np.float32)
        self.dones     = np.zeros((self.n_steps_per_env, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps_per_env, n_envs), dtype=np.float32)
        self.values    = np.zeros((self.n_steps_per_env, n_envs), dtype=np.float32)

    def add(self, obs, action, reward, done, log_prob, value):
        """Accept arrays of shape (n_envs,) / (n_envs, obs_size) and store at current ptr."""
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1
        if self.ptr >= self.n_steps_per_env:
            self.full = True

    def is_full(self):
        return self.full

    def clear(self):
        self.ptr  = 0
        self.full = False

    def get_tensors(self, device):
        """Return flattened 1D tensors for mini-batch updates."""
        total = self.total_steps
        return (
            torch.tensor(self.obs.reshape(total, -1),         dtype=torch.float32).to(device),
            torch.tensor(self.actions.reshape(total),         dtype=torch.int64).to(device),
            torch.tensor(self.rewards.reshape(total),         dtype=torch.float32).to(device),
            torch.tensor(self.dones.reshape(total),           dtype=torch.float32).to(device),
            torch.tensor(self.log_probs.reshape(total),       dtype=torch.float32).to(device),
            torch.tensor(self.values.reshape(total),          dtype=torch.float32).to(device),
        )

    def get_rewards_dones_2d(self, device):
        """Return (n_steps_per_env, n_envs) tensors for per-env return computation."""
        return (
            torch.tensor(self.rewards, dtype=torch.float32).to(device),
            torch.tensor(self.dones,   dtype=torch.float32).to(device),
        )


# Calculates the "target" return should be for each step (Discounted Reward)
# Operates on 2D inputs (n_steps_per_env, n_envs) for correct per-env computation
def compute_returns(rewards_2d, dones_2d, last_values, gamma):
    n_steps, n_envs = rewards_2d.shape
    returns = torch.zeros_like(rewards_2d)

    # Estimated future value if the buffer ended here — one per env
    R = last_values  # (n_envs,)

    # Iterate backwards through all the steps taken and accumulate rewards
    for t in reversed(range(n_steps)):
        R = rewards_2d[t] + gamma * R * (1.0 - dones_2d[t])
        returns[t] = R
    return returns.reshape(-1)  # flatten to (n_steps_per_env * n_envs,)


class PPO(BaseAgent):
    def __init__(self, model, cfg, n_envs=1):
        self.model     = model
        self.cfg       = cfg
        self.n_envs    = n_envs
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Adam optimizer with a small epsilon for better numerical stability
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)
        self.buffer    = RolloutBuffer(cfg.N_STEPS, cfg.INPUT_SIZE, n_envs=n_envs)
        print("PPO initialized on device:", self.device)
        if n_envs > 1:
            print("  Parallel envs: {0} | Steps per env: {1}".format(
                n_envs, self.buffer.n_steps_per_env))

    def collect_steps(self, envs, obs_all):
        """Batched collection: single GPU forward pass for all envs, then step each env."""
        n_envs = len(envs)
        # Stack all observations for a single forward pass
        obs_t = torch.tensor(obs_all, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            dist     = self.model.get_distribution(obs_t)
            actions  = dist.sample()                    # (n_envs,)
            log_probs = dist.log_prob(actions)          # (n_envs,)
            values   = self.model.get_value(obs_t).squeeze(-1)  # (n_envs,)

        actions_np   = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np    = values.cpu().numpy()

        next_obs_all = np.zeros_like(obs_all)
        rewards      = np.zeros(n_envs, dtype=np.float32)
        dones        = np.zeros(n_envs, dtype=np.float32)
        infos        = [None] * n_envs

        for i, env in enumerate(envs):
            next_obs, reward, done, info = env.step(int(actions_np[i]))
            next_obs_all[i] = next_obs
            rewards[i]      = reward
            dones[i]        = float(done)
            infos[i]        = info

        self.buffer.add(obs_all, actions_np, rewards, dones, log_probs_np, values_np)
        return next_obs_all, rewards, dones, infos

    # samples random actions in order to explore
    def collect_step(self, env, obs):
        """Single-env wrapper around collect_steps for backward compat."""
        obs_all = obs[np.newaxis]  # (1, obs_size)
        next_obs_all, rewards, dones, infos = self.collect_steps([env], obs_all)
        return next_obs_all[0], rewards[0], bool(dones[0]), infos[0]

    def buffer_full(self):
        return self.buffer.is_full()

    # PPO Update Step
    def update(self, last_obs):

        # Bootstrapping step: predict the future rewards from this step
        # last_obs is (n_envs, obs_size) for multi-env, (obs_size,) for single-env
        last_obs_2d = np.atleast_2d(last_obs)
        with torch.no_grad():
            last_vals = self.model.get_value(
                torch.tensor(last_obs_2d, dtype=torch.float32).to(self.device)
            ).squeeze(-1)  # (n_envs,)

        # Read from RolloutBuffer
        obs_t, actions_t, rewards_flat_t, dones_flat_t, old_log_probs_t, values_t = \
            self.buffer.get_tensors(self.device)

        # Per-env return computation using 2D arrays
        rewards_2d, dones_2d = self.buffer.get_rewards_dones_2d(self.device)
        returns = compute_returns(rewards_2d, dones_2d, last_vals, self.cfg.GAMMA).to(self.device)

        # Advantage indicates whether the agent did something better/worse than its current strategy
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = self.buffer.total_steps
        indices = np.arange(total_steps)
        total_policy_loss = total_value_loss = total_entropy = n_updates = 0

        for _ in range(self.cfg.N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_steps, self.cfg.BATCH_SIZE):
                idx = indices[start:start + self.cfg.BATCH_SIZE]

                # reevaluate old actions using the new policy
                new_log_probs, new_values, entropy = self.model.evaluate_actions(
                    obs_t[idx], actions_t[idx]
                )

                # PPO Clipped Objective, compares new to old policy
                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])

                # prevent the policy from changing too much (Clipped)
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.cfg.CLIP_EPS, 1 + self.cfg.CLIP_EPS) * advantages[idx]

                # Actor Loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss
                value_loss = nn.functional.mse_loss(new_values, returns[idx])

                # Total Loss = Policy + Critic - Entropy
                loss = policy_loss + self.cfg.VALUE_COEF * value_loss - self.cfg.ENTROPY_COEF * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1


        self.buffer.clear()
        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss":  total_value_loss / max(n_updates, 1),
            "entropy":     total_entropy / max(n_updates, 1),
        }

    # For inference choose an action
    def select_action(self, obs, greedy=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if greedy:
                return self.model(obs_t)[0].argmax(dim=-1).item()
            return self.model.get_distribution(obs_t).sample().item()
