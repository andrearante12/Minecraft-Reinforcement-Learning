"""
algos/ppo.py
------------
Proximal Policy Optimization — on-policy, learns data from data collected by the most recent
version of the policy

Improvements over vanilla PPO:
  - GAE (Generalized Advantage Estimation) instead of Monte Carlo returns
  - Value function clipping
  - Observation normalization (running mean/std)
  - Reward normalization (running std, no mean subtraction)
  - Learning rate linear decay
  - Entropy coefficient linear decay
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algos.base_agent import BaseAgent


class RunningMeanStd:
    """Tracks running mean and variance using Welford's online algorithm."""
    def __init__(self, shape=()):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean  = batch.mean(axis=0)
        batch_var   = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2  = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var   = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]


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

    def get_values_2d(self, device):
        """Return (n_steps_per_env, n_envs) tensor for GAE computation."""
        return torch.tensor(self.values, dtype=torch.float32).to(device)


def compute_gae(rewards_2d, values_2d, dones_2d, last_values, gamma, gae_lambda):
    """GAE advantage estimation. Returns (advantages, returns) both flattened."""
    n_steps, n_envs = rewards_2d.shape
    advantages = torch.zeros_like(rewards_2d)
    gae = torch.zeros(n_envs, device=rewards_2d.device)

    for t in reversed(range(n_steps)):
        next_values = last_values if t == n_steps - 1 else values_2d[t + 1]
        delta = rewards_2d[t] + gamma * next_values * (1.0 - dones_2d[t]) - values_2d[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones_2d[t]) * gae
        advantages[t] = gae

    returns = advantages + values_2d
    return advantages.reshape(-1), returns.reshape(-1)


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

        # Normalization
        self.obs_rms    = RunningMeanStd(shape=(cfg.INPUT_SIZE,)) if cfg.OBS_NORM else None
        self.reward_rms = RunningMeanStd(shape=()) if cfg.REWARD_NORM else None

        # LR & entropy scheduling
        self.initial_lr           = cfg.LR
        self.initial_entropy_coef = cfg.ENTROPY_COEF
        self.current_lr           = cfg.LR
        self.current_entropy_coef = cfg.ENTROPY_COEF

        print("PPO initialized on device:", self.device)
        if n_envs > 1:
            print("  Parallel envs: {0} | Steps per env: {1}".format(
                n_envs, self.buffer.n_steps_per_env))
        if self.obs_rms:
            print("  Obs normalization: ON")
        if self.reward_rms:
            print("  Reward normalization: ON")
        if cfg.LR_DECAY:
            print("  LR decay: {0} -> {1}".format(cfg.LR, cfg.LR_END))
        if cfg.ENTROPY_DECAY:
            print("  Entropy decay: {0} -> {1}".format(cfg.ENTROPY_COEF, cfg.ENTROPY_COEF_END))

    def set_progress(self, fraction):
        """Called by train.py each episode. fraction = episode / total_episodes (0->1)."""
        fraction = max(0.0, min(1.0, fraction))

        if self.cfg.LR_DECAY:
            self.current_lr = self.initial_lr + (self.cfg.LR_END - self.initial_lr) * fraction
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr

        if self.cfg.ENTROPY_DECAY:
            self.current_entropy_coef = (
                self.initial_entropy_coef
                + (self.cfg.ENTROPY_COEF_END - self.initial_entropy_coef) * fraction
            )

    def _normalize_obs(self, obs):
        """Normalize observations using running stats (in-place update + normalize)."""
        if self.obs_rms:
            return self.obs_rms.normalize(obs, self.cfg.NORM_CLIP)
        return obs

    def collect_steps(self, envs, obs_all):
        """Batched collection: single GPU forward pass for all envs, then step each env."""
        n_envs = len(envs)

        # Update obs normalizer with raw obs, then normalize for forward pass
        if self.obs_rms:
            self.obs_rms.update(obs_all)
            obs_normed = self.obs_rms.normalize(obs_all, self.cfg.NORM_CLIP)
        else:
            obs_normed = obs_all

        # Stack all observations for a single forward pass
        obs_t = torch.tensor(obs_normed, dtype=torch.float32).to(self.device)

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

        # Normalize rewards (std only, no mean subtraction)
        if self.reward_rms:
            self.reward_rms.update(rewards)
            rewards_normed = rewards / np.sqrt(self.reward_rms.var + 1e-8)
            rewards_normed = np.clip(rewards_normed, -self.cfg.NORM_CLIP, self.cfg.NORM_CLIP)
        else:
            rewards_normed = rewards

        # Store normalized obs and rewards in buffer
        self.buffer.add(obs_normed, actions_np, rewards_normed, dones, log_probs_np, values_np)

        # Normalize next obs for return to caller
        if self.obs_rms:
            next_obs_all_normed = self.obs_rms.normalize(next_obs_all, self.cfg.NORM_CLIP)
        else:
            next_obs_all_normed = next_obs_all

        # Return raw next_obs (train.py tracks raw obs, we normalize at the start of collect_steps)
        # But return raw rewards for episode tracking
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
        if self.obs_rms:
            last_obs_2d = self.obs_rms.normalize(last_obs_2d, self.cfg.NORM_CLIP)
        with torch.no_grad():
            last_vals = self.model.get_value(
                torch.tensor(last_obs_2d, dtype=torch.float32).to(self.device)
            ).squeeze(-1)  # (n_envs,)

        # Read from RolloutBuffer
        obs_t, actions_t, rewards_flat_t, dones_flat_t, old_log_probs_t, values_t = \
            self.buffer.get_tensors(self.device)

        # Per-env GAE computation using 2D arrays
        rewards_2d, dones_2d = self.buffer.get_rewards_dones_2d(self.device)
        values_2d = self.buffer.get_values_2d(self.device)
        advantages, returns = compute_gae(
            rewards_2d, values_2d, dones_2d, last_vals,
            self.cfg.GAMMA, self.cfg.GAE_LAMBDA
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
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

                # Critic Loss (clipped value function)
                value_clipped = values_t[idx] + torch.clamp(
                    new_values - values_t[idx], -self.cfg.CLIP_EPS, self.cfg.CLIP_EPS
                )
                vl_unclipped = (new_values - returns[idx]) ** 2
                vl_clipped   = (value_clipped - returns[idx]) ** 2
                value_loss   = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                # Total Loss = Policy + Critic - Entropy
                loss = policy_loss + self.cfg.VALUE_COEF * value_loss - self.current_entropy_coef * entropy

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
            "lr":          self.current_lr,
            "entropy_coef": self.current_entropy_coef,
        }

    # For inference choose an action
    def select_action(self, obs, greedy=False):
        if self.obs_rms:
            obs = self.obs_rms.normalize(obs, self.cfg.NORM_CLIP)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if greedy:
                return self.model(obs_t)[0].argmax(dim=-1).item()
            return self.model.get_distribution(obs_t).sample().item()

    def _extra_state(self):
        state = {
            "current_lr": self.current_lr,
            "current_entropy_coef": self.current_entropy_coef,
        }
        if self.obs_rms:
            state["obs_rms"] = self.obs_rms.state_dict()
        if self.reward_rms:
            state["reward_rms"] = self.reward_rms.state_dict()
        return state

    def _load_extra_state(self, state):
        if "current_lr" in state:
            self.current_lr = state["current_lr"]
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr
        if "current_entropy_coef" in state:
            self.current_entropy_coef = state["current_entropy_coef"]
        if "obs_rms" in state and self.obs_rms:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if "reward_rms" in state and self.reward_rms:
            self.reward_rms.load_state_dict(state["reward_rms"])
