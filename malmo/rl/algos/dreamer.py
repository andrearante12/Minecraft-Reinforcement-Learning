"""
algos/dreamer.py
----------------
Model-based RL agent (Dreamer-style) — implements BaseAgent.

Idea: instead of learning the policy from millions of expensive real Malmo
steps, the agent collects a modest number of real transitions, fits a compact
world model (models/world_model.py) to them, and then trains the actor-critic
almost entirely inside the world model's "imagination". Each real Malmo step is
a blocking ~0.15s hold, so replacing real steps with cheap imagined ones is a
large wall-clock win.

Loop per update():
  (a) WORLD-MODEL LEARNING — sample real transitions from the replay buffer and
      fit the dynamics/reward/continuation model.
  (b) BEHAVIOUR LEARNING IN IMAGINATION — sample real start states, roll the
      (frozen) world model IMAG_HORIZON steps under the current actor, compute
      Dreamer λ-returns, and update the actor (REINFORCE / score-function, since
      actions are discrete) and critic (regression to λ-returns).

Checkpoint compatibility: self.model is the ActorCritic (so policy checkpoints
stay byte-compatible with PPO/DQN/BC — warm-start works both ways). The world
model + its optimizer are persisted via _extra_state().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algos.base_agent import BaseAgent
from algos.replay_buffer import ReplayBuffer
from models.world_model import WorldModel


class Dreamer(BaseAgent):
    def __init__(self, model, cfg, n_envs=1):
        self.model   = model            # ActorCritic (policy + value)
        self.cfg     = cfg
        self.n_envs  = n_envs
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)

        # World model + its own optimizer
        self.world_model   = WorldModel(cfg).to(self.device)
        self.wm_optimizer  = optim.Adam(self.world_model.parameters(), lr=cfg.WM_LR, eps=1e-5)

        self.buffer        = ReplayBuffer(cfg.WM_BUFFER_CAPACITY)
        self.steps_done    = 0

        # Actor entropy schedule (mirrors PPO's optional decay)
        self.initial_entropy_coef = cfg.IMAG_ENTROPY_COEF
        self.current_entropy_coef = cfg.IMAG_ENTROPY_COEF
        self.initial_lr           = cfg.LR
        self.current_lr           = cfg.LR

        print("Dreamer initialized on device:", self.device)
        print("  World model params: {0}".format(
            sum(p.numel() for p in self.world_model.parameters())))
        print("  Imagination horizon: {0} | batch: {1} | lambda: {2}".format(
            cfg.IMAG_HORIZON, cfg.IMAG_BATCH, cfg.IMAG_LAMBDA))
        print("  Learning starts at {0} real transitions".format(cfg.LEARNING_STARTS))

    # ── Data collection ─────────────────────────────────────────────────────────
    def collect_step(self, env, obs):
        """Act with the stochastic actor; store the real transition."""
        self.steps_done += 1
        action = self.select_action(obs, greedy=False)
        next_obs, reward, done, info = env.step(action)
        self.buffer.add(obs, action, reward, next_obs, done)
        return next_obs, reward, done, info

    # collect_steps: inherit BaseAgent default (loops collect_step). All envs feed
    # one shared buffer — desirable for diverse world-model training data.

    def buffer_full(self):
        return (len(self.buffer) >= self.cfg.LEARNING_STARTS
                and self.steps_done % self.cfg.REAL_TRAIN_EVERY == 0)

    def select_action(self, obs, greedy=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(obs_t)
            if greedy:
                return logits.argmax(dim=-1).item()
            return Categorical(logits=logits).sample().item()

    # ── Update: train world model, then train behaviour in imagination ──────────
    def update(self, last_obs=None):
        logs = {}
        for _ in range(self.cfg.WM_GRAD_STEPS):
            logs.update(self._train_world_model())
        for _ in range(self.cfg.IMAG_UPDATES):
            logs.update(self._imagine_and_train())
        logs["lr"] = self.current_lr
        logs["entropy_coef"] = self.current_entropy_coef
        return logs

    def _train_world_model(self):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.WM_BATCH_SIZE)
        obs_t      = torch.as_tensor(obs,      dtype=torch.float32, device=self.device)
        actions_t  = torch.as_tensor(actions,  dtype=torch.int64,   device=self.device)
        rewards_t  = torch.as_tensor(rewards,  dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t    = torch.as_tensor(dones,    dtype=torch.float32, device=self.device)

        loss, metrics = self.world_model.loss(obs_t, actions_t, rewards_t, next_obs_t, dones_t)

        self.wm_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.WM_MAX_GRAD_NORM)
        self.wm_optimizer.step()
        return metrics

    def _imagine_and_train(self):
        """Roll the frozen world model from real start states; train actor-critic."""
        H       = self.cfg.IMAG_HORIZON
        gamma   = self.cfg.GAMMA
        lam     = self.cfg.IMAG_LAMBDA

        starts = self.buffer.sample_starts(self.cfg.IMAG_BATCH)
        s = torch.as_tensor(starts, dtype=torch.float32, device=self.device)

        log_probs, entropies, values = [], [], []
        rewards, conts = [], []

        for _ in range(H):
            logits, value = self.model(s)              # actor-critic forward (grad)
            dist = Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            values.append(value.squeeze(-1))
            with torch.no_grad():                      # world model is frozen here
                s, r, cont = self.world_model.predict(s, a)
            rewards.append(r)
            conts.append(cont)

        with torch.no_grad():
            _, last_value = self.model(s)
        last_value = last_value.squeeze(-1)

        log_probs = torch.stack(log_probs)             # (H, B)
        entropies = torch.stack(entropies)             # (H, B)
        values    = torch.stack(values)                # (H, B)  — requires grad
        rewards   = torch.stack(rewards)               # (H, B)  — detached
        conts     = torch.stack(conts)                 # (H, B)  — detached

        # Dreamer λ-returns (value targets treated as constants → detach values)
        values_det = values.detach()
        returns = torch.zeros_like(values_det)
        g = last_value
        for t in reversed(range(H)):
            next_v = values_det[t + 1] if t < H - 1 else last_value
            g = rewards[t] + gamma * conts[t] * ((1.0 - lam) * next_v + lam * g)
            returns[t] = g

        advantage = (returns - values_det)             # detached → REINFORCE baseline

        actor_loss  = -(log_probs * advantage).mean() - self.current_entropy_coef * entropies.mean()
        critic_loss = 0.5 * ((values - returns) ** 2).mean()
        loss = actor_loss + self.cfg.IMAG_VALUE_COEF * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropies.mean().item(),
            "imag_return": returns.mean().item(),
            "imag_reward": rewards.mean().item(),
        }

    # ── Optional LR / entropy schedule (train.py calls this if present) ─────────
    def set_progress(self, fraction):
        fraction = max(0.0, min(1.0, fraction))
        if getattr(self.cfg, "LR_DECAY", False):
            self.current_lr = self.initial_lr + (self.cfg.LR_END - self.initial_lr) * fraction
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr
        if getattr(self.cfg, "ENTROPY_DECAY", False):
            end = getattr(self.cfg, "ENTROPY_COEF_END", self.initial_entropy_coef)
            self.current_entropy_coef = self.initial_entropy_coef + (end - self.initial_entropy_coef) * fraction

    # ── Checkpointing ───────────────────────────────────────────────────────────
    def _extra_state(self):
        return {
            "world_model_state": self.world_model.state_dict(),
            "wm_optimizer_state": self.wm_optimizer.state_dict(),
            "steps_done": self.steps_done,
            "current_lr": self.current_lr,
            "current_entropy_coef": self.current_entropy_coef,
        }

    def _load_extra_state(self, state):
        if "world_model_state" in state:
            self.world_model.load_state_dict(state["world_model_state"])
        if "wm_optimizer_state" in state:
            self.wm_optimizer.load_state_dict(state["wm_optimizer_state"])
        self.steps_done = state.get("steps_done", 0)
        self.current_lr = state.get("current_lr", self.initial_lr)
        self.current_entropy_coef = state.get("current_entropy_coef", self.initial_entropy_coef)
