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

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algos.base_agent import BaseAgent
from algos.replay_buffer import ReplayBuffer
from models.world_model import WorldModel, WorldModelEnsemble


class Dreamer(BaseAgent):
    def __init__(self, model, cfg, n_envs=1):
        self.model   = model            # ActorCritic (policy + value)
        self.cfg     = cfg
        self.n_envs  = n_envs
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)

        # World model + its own optimizer. An ensemble (K>1) enables the
        # aleatoric/epistemic uncertainty decomposition; K=1 is the single
        # deterministic model (byte-identical to the validated Phase-0 path).
        self.is_ensemble   = getattr(cfg, "ENSEMBLE_SIZE", 1) > 1
        self.world_model   = (WorldModelEnsemble(cfg) if self.is_ensemble
                              else WorldModel(cfg)).to(self.device)
        self.wm_optimizer  = optim.Adam(self.world_model.parameters(), lr=cfg.WM_LR, eps=1e-5)

        # Uncertainty consumers (both require an ensemble to estimate epistemic).
        self.horizon_gating = self.is_ensemble and getattr(cfg, "WM_HORIZON_GATING", False)
        self.trust_beta     = getattr(cfg, "WM_TRUST_BETA", 0.5)
        self.intrinsic_coef = getattr(cfg, "INTRINSIC_COEF", 0.0) if self.is_ensemble else 0.0
        self._epi_norm      = 1.0   # EMA normaliser keeping intrinsic reward ~O(coef)

        self.buffer        = ReplayBuffer(cfg.WM_BUFFER_CAPACITY)
        self.steps_done    = 0

        # DreamerFD-style demo integration
        self.demo_buffer     = None
        self._demo_bc_coef   = getattr(cfg, "DEMO_BC_COEF", 0.0)
        demo_path            = getattr(cfg, "DEMO_PATH", None)
        if demo_path and os.path.exists(demo_path):
            self.demo_buffer = self._load_demo_buffer(demo_path)
            print("  Demo buffer: {0} transitions from {1}".format(
                len(self.demo_buffer), demo_path))
        elif demo_path:
            print("  WARNING: demo path not found: {0}".format(demo_path))

        # Actor entropy schedule (mirrors PPO's optional decay)
        self.initial_entropy_coef = cfg.IMAG_ENTROPY_COEF
        self.current_entropy_coef = cfg.IMAG_ENTROPY_COEF
        self.initial_lr           = cfg.LR
        self.current_lr           = cfg.LR

        print("Dreamer initialized on device:", self.device)
        print("  World model: {0}  params: {1}".format(
            "ensemble x{0} (probabilistic={1})".format(
                cfg.ENSEMBLE_SIZE, getattr(cfg, "WM_PROBABILISTIC", False))
            if self.is_ensemble else "single deterministic",
            sum(p.numel() for p in self.world_model.parameters())))
        print("  Uncertainty: horizon_gating={0} (beta={1}) intrinsic_coef={2} ({3})".format(
            self.horizon_gating, self.trust_beta, self.intrinsic_coef,
            getattr(cfg, "INTRINSIC_KIND", "epistemic")))
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
        logs.update(self._demo_bc_update())
        logs["lr"] = self.current_lr
        logs["entropy_coef"] = self.current_entropy_coef
        return logs

    def _batch_tensors(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        return (
            torch.as_tensor(obs,      dtype=torch.float32, device=self.device),
            torch.as_tensor(actions,  dtype=torch.int64,   device=self.device),
            torch.as_tensor(rewards,  dtype=torch.float32, device=self.device),
            torch.as_tensor(next_obs, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones,    dtype=torch.float32, device=self.device),
        )

    def _load_demo_buffer(self, path):
        """Load demo JSON into a ReplayBuffer.

        Supports two formats:
          - full: steps have "reward", "next_obs", "done" keys
          - BC-only (legacy): steps have only "obs" and "action"; next_obs is
            inferred from the next step's obs, reward=0, done=False/last-step.
        """
        with open(path) as f:
            data = json.load(f)
        buf = ReplayBuffer(capacity=200_000)
        for ep in data["episodes"]:
            steps = ep["steps"]
            for i, step in enumerate(steps):
                obs = np.array(step["obs"], dtype=np.float32)
                action = int(step["action"])
                reward = float(step.get("reward", 0.0))
                if "next_obs" in step:
                    next_obs = np.array(step["next_obs"], dtype=np.float32)
                elif i + 1 < len(steps):
                    next_obs = np.array(steps[i + 1]["obs"], dtype=np.float32)
                else:
                    next_obs = np.zeros_like(obs)
                done = bool(step.get("done", i + 1 == len(steps)))
                buf.add(obs, action, reward, next_obs, done)
        return buf

    def _train_world_model(self):
        # Each ensemble member fits its OWN independently sampled minibatch
        # (bootstrap) so members diversify → nonzero epistemic uncertainty.
        # For a single model this is exactly one batch / one loss as before.
        # With a demo buffer, a small fraction (DEMO_WM_FRACTION) of each
        # minibatch is drawn from expert transitions.
        members = self.world_model.members if self.is_ensemble else [self.world_model]
        demo_frac = getattr(self.cfg, "DEMO_WM_FRACTION", 0.1)
        self.wm_optimizer.zero_grad()
        losses, agg = [], None
        for m in members:
            batch_size = self.cfg.WM_BATCH_SIZE
            if self.demo_buffer and len(self.demo_buffer) > 0:
                n_demo = max(1, int(batch_size * demo_frac))
                n_real = batch_size - n_demo
                real_b = self.buffer.sample(n_real)
                demo_b = self.demo_buffer.sample(n_demo)
                obs      = np.concatenate([real_b[0], demo_b[0]], axis=0)
                actions  = np.concatenate([real_b[1], demo_b[1]], axis=0)
                rewards  = np.concatenate([real_b[2], demo_b[2]], axis=0)
                next_obs = np.concatenate([real_b[3], demo_b[3]], axis=0)
                dones    = np.concatenate([real_b[4], demo_b[4]], axis=0)
                tensors  = self._batch_tensors((obs, actions, rewards, next_obs, dones))
            else:
                tensors = self._batch_tensors(self.buffer.sample(batch_size))
            loss, metrics = m.loss(*tensors)
            losses.append(loss)
            agg = metrics if agg is None else {k: agg[k] + metrics[k] for k in agg}
        total = torch.stack(losses).sum()
        total.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.WM_MAX_GRAD_NORM)
        self.wm_optimizer.step()
        n = len(members)
        return {k: v / n for k, v in agg.items()}

    def _imagine_and_train(self):
        """Roll the frozen world model from real start states; train actor-critic.

        With an ensemble, each imagined step also yields the (aleatoric, epistemic)
        uncertainty on the target. Two consumers act on it:
          • horizon gating — a trust factor τ_t = exp(-β·Σ uncertainty) folded into
            the GAMMA·continuation discount, so returns stop crediting diverged
            far-future dreams;
          • exploration — an epistemic-ONLY intrinsic reward (never aleatoric, to
            avoid chasing the pig's irreducible noise).
        """
        H       = self.cfg.IMAG_HORIZON
        gamma   = self.cfg.GAMMA
        lam     = self.cfg.IMAG_LAMBDA

        # Blend real and demo start states so imagination explores the expert's
        # distribution early in training (when demo BC coef is large).
        batch = self.cfg.IMAG_BATCH
        if self.demo_buffer and len(self.demo_buffer) > 0:
            start_frac = getattr(self.cfg, "DEMO_START_FRACTION", 0.3)
            n_demo = max(1, int(batch * start_frac))
            n_real = batch - n_demo
            real_s  = self.buffer.sample_starts(n_real)
            demo_s  = self.demo_buffer.sample_starts(n_demo)
            starts  = np.concatenate([real_s, demo_s], axis=0)
        else:
            starts = self.buffer.sample_starts(batch)
        s = torch.as_tensor(starts, dtype=torch.float32, device=self.device)
        B = s.shape[0]

        log_probs, entropies, values = [], [], []
        rewards, conts, trusts       = [], [], []
        ales, epis, intrinsics       = [], [], []

        U = torch.zeros(B, device=self.device)         # cumulative uncertainty
        for _ in range(H):
            logits, value = self.model(s)              # actor-critic forward (grad)
            dist = Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            values.append(value.squeeze(-1))
            with torch.no_grad():                      # world model is frozen here
                if self.is_ensemble:
                    s, r, cont, ale, epi = self.world_model.predict_with_uncertainty(s, a)
                else:
                    s, r, cont = self.world_model.predict(s, a)
                    ale = epi = torch.zeros(B, device=self.device)

            # Horizon gating: trust the future less as uncertainty accumulates.
            U = U + ale + epi
            trust = torch.exp(-self.trust_beta * U) if self.horizon_gating else torch.ones_like(U)

            # Exploration: epistemic-only intrinsic reward (noisy-TV guard).
            intr = (self.intrinsic_coef * epi / (self._epi_norm + 1e-8)
                    if self.intrinsic_coef > 0.0 else torch.zeros_like(epi))

            rewards.append(r + intr)
            conts.append(cont)
            trusts.append(trust)
            ales.append(ale); epis.append(epi); intrinsics.append(intr)

        with torch.no_grad():
            _, last_value = self.model(s)
        last_value = last_value.squeeze(-1)

        log_probs = torch.stack(log_probs)             # (H, B)
        entropies = torch.stack(entropies)             # (H, B)
        values    = torch.stack(values)                # (H, B)  — requires grad
        rewards   = torch.stack(rewards)               # (H, B)  — detached
        conts     = torch.stack(conts)                 # (H, B)  — detached
        trusts    = torch.stack(trusts)                # (H, B)  — detached

        # Update the intrinsic-reward normaliser (EMA of epistemic magnitude).
        if self.intrinsic_coef > 0.0:
            self._epi_norm = 0.99 * self._epi_norm + 0.01 * torch.stack(epis).mean().item()

        # Dreamer λ-returns; the trust factor rides inside the discount.
        values_det = values.detach()
        returns = torch.zeros_like(values_det)
        g = last_value
        for t in reversed(range(H)):
            next_v = values_det[t + 1] if t < H - 1 else last_value
            disc = gamma * conts[t] * trusts[t]
            g = rewards[t] + disc * ((1.0 - lam) * next_v + lam * g)
            returns[t] = g

        advantage = (returns - values_det)             # detached → REINFORCE baseline

        actor_loss  = -(log_probs * advantage).mean() - self.current_entropy_coef * entropies.mean()
        critic_loss = 0.5 * ((values - returns) ** 2).mean()
        loss = actor_loss + self.cfg.IMAG_VALUE_COEF * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()

        # eff_horizon ≈ number of imagined steps still trusted (Σ τ_t), ≤ H.
        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropies.mean().item(),
            "imag_return": returns.mean().item(),
            "imag_reward": rewards.mean().item(),
            "aleatoric":   torch.stack(ales).mean().item(),
            "epistemic":   torch.stack(epis).mean().item(),
            "eff_horizon": trusts.sum(dim=0).mean().item(),
            "intrinsic_r": torch.stack(intrinsics).mean().item(),
        }

    def _demo_bc_update(self):
        """Behavioural-cloning cross-entropy anchor on demo (obs, action) pairs.

        Decays with training progress via _demo_bc_coef so the actor can improve
        beyond the demonstrator without being permanently anchored.
        """
        if self._demo_bc_coef <= 0.0 or not self.demo_buffer or len(self.demo_buffer) == 0:
            return {}
        batch = self.demo_buffer.sample(min(len(self.demo_buffer), self.cfg.WM_BATCH_SIZE))
        obs_t, actions_t, _, _, _ = self._batch_tensors(batch)
        logits, _ = self.model(obs_t)
        bc_loss = nn.CrossEntropyLoss()(logits, actions_t)
        self.optimizer.zero_grad()
        (self._demo_bc_coef * bc_loss).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()
        return {"bc_loss": bc_loss.item()}

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
        # Decay BC anchor so the agent can eventually improve beyond the demonstrator.
        if self._demo_bc_coef > 0.0:
            initial_bc = getattr(self.cfg, "DEMO_BC_COEF", self._demo_bc_coef)
            self._demo_bc_coef = initial_bc * (1.0 - fraction)

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
