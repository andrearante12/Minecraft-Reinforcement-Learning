"""
models/world_model.py
---------------------
Compact, observation-space dynamics model for model-based RL (Dreamer-style).

Given (obs, action) it predicts:
  - next observation  — proprio & goal as RESIDUAL DELTAS added to the current
                        obs (smooth over a 0.15s action hold → easy/stable to
                        learn, no-op identity is trivial); voxel grid as binary
                        occupancy logits (sigmoid → [0,1]).
  - reward            — scalar (MSE).
  - continuation      — P(not done) logit (BCE); discount in imagination is
                        GAMMA * continuation, so episode termination is handled
                        smoothly without hard masking.

Because MalmoRL observations are already low-dimensional structured state
vectors, the model predicts directly in observation space. Imagined
observations therefore have the SAME shape/semantics as real ones and feed
straight into the existing ActorCritic with no decoder.

The encoder reuses the exact multi-stream builders from actor_critic.py, and
the config-driven obs split (PROPRIOCEPTION_SIZE | GOAL_DELTA_SIZE | GRID_SIZE)
— so it works unchanged for parkour (6/3/150), bridging (17/3/200), and the
hunting env (proprio/target/voxel).

── Uncertainty-aware variant (default OFF) ────────────────────────────────────
When cfg.WM_PROBABILISTIC is True the GOAL/target stream head emits a Gaussian
(mean_delta, log_var_delta) instead of a point delta, trained with Gaussian NLL.
A WorldModelEnsemble of K such members then yields a principled decomposition of
predictive uncertainty on the moving target (e.g. a pig):

    aleatoric = mean_k( var_k )        # irreducible noise (the pig's randomness)
    epistemic = var_k( mean_k )        # model ignorance (shrinks with data)

Both default off (WM_PROBABILISTIC False, ENSEMBLE_SIZE 1) → this module reduces
exactly to the deterministic single-model path used for Phase-0/1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor_critic import _make_stream


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.proprio_size = cfg.PROPRIOCEPTION_SIZE
        self.goal_size    = cfg.GOAL_DELTA_SIZE
        self.voxel_size   = cfg.GRID_SIZE
        self._proprio_end = self.proprio_size
        self._goal_end    = self.proprio_size + self.goal_size

        # Uncertainty knobs (all default OFF → deterministic point prediction).
        self.probabilistic = getattr(cfg, "WM_PROBABILISTIC", False)
        self.min_logvar    = getattr(cfg, "WM_MIN_LOGVAR", -8.0)
        self.max_logvar    = getattr(cfg, "WM_MAX_LOGVAR", 2.0)

        ph = cfg.WM_PROPRIO_HIDDEN
        gh = cfg.WM_GOAL_HIDDEN
        vh = cfg.WM_VOXEL_HIDDEN

        # ── Encoder: same 3-stream idea as ActorCritic ──────────────────────────
        self.proprio_stream = _make_stream(self.proprio_size, ph)
        self.goal_stream    = _make_stream(self.goal_size, gh)
        self.voxel_stream   = _make_stream(self.voxel_size, vh)

        # ── Action conditioning ─────────────────────────────────────────────────
        self.act_embed = nn.Embedding(cfg.N_ACTIONS, cfg.WM_ACTION_EMBED)

        # ── Shared trunk: (encoded obs, action) -> joint feature ────────────────
        trunk_in = ph + gh + vh + cfg.WM_ACTION_EMBED
        self.trunk = _make_stream(trunk_in, cfg.WM_HIDDEN)

        # ── Prediction heads ────────────────────────────────────────────────────
        h = cfg.WM_HIDDEN
        self.proprio_head = nn.Linear(h, self.proprio_size)   # residual delta
        # Goal head: point delta (deterministic) or (mean, log_var) (probabilistic)
        goal_out = self.goal_size * (2 if self.probabilistic else 1)
        self.goal_head    = nn.Linear(h, goal_out)
        if self.probabilistic:
            # Start the log-var output near-constant and modest so an UNTRAINED
            # model doesn't report huge aleatoric uncertainty (which would make
            # horizon gating strangle imagination before the model has learned).
            # Predicted variance then LEARNS UP from the real transition noise.
            init_logvar = getattr(cfg, "WM_INIT_LOGVAR", -2.0)
            with torch.no_grad():
                self.goal_head.weight[self.goal_size:] *= 0.01
                self.goal_head.bias[self.goal_size:].fill_(init_logvar)
        self.voxel_head   = nn.Linear(h, self.voxel_size)     # occupancy logits
        self.reward_head  = _make_head_scalar(h, cfg.WM_HEAD_HIDDEN)
        self.cont_head    = _make_head_scalar(h, cfg.WM_HEAD_HIDDEN)  # P(not done) logit

    # ── Forward / prediction ────────────────────────────────────────────────────
    def _encode(self, obs):
        proprio = obs[:, :self._proprio_end]
        goal    = obs[:, self._proprio_end:self._goal_end]
        voxel   = obs[:, self._goal_end:]
        return torch.cat([
            self.proprio_stream(proprio),
            self.goal_stream(goal),
            self.voxel_stream(voxel),
        ], dim=-1)

    def _goal_out(self, h):
        """Return (mean_delta, log_var_delta). log_var is None when deterministic."""
        raw = self.goal_head(h)
        if self.probabilistic:
            mean   = raw[:, :self.goal_size]
            logvar = torch.clamp(raw[:, self.goal_size:], self.min_logvar, self.max_logvar)
            return mean, logvar
        return raw, None

    def _heads(self, obs, action):
        """All raw head outputs as a dict (single forward pass)."""
        feat = self._encode(obs)
        a    = self.act_embed(action)
        h    = self.trunk(torch.cat([feat, a], dim=-1))
        d_goal_mean, d_goal_logvar = self._goal_out(h)
        return {
            "d_proprio":     self.proprio_head(h),
            "d_goal_mean":   d_goal_mean,
            "d_goal_logvar": d_goal_logvar,          # None if deterministic
            "voxel_logits":  self.voxel_head(h),
            "reward":        self.reward_head(h).squeeze(-1),
            "cont_logit":    self.cont_head(h).squeeze(-1),
        }

    def forward(self, obs, action):
        """Back-compat 5-tuple: (d_proprio, d_goal_mean, voxel_logits, reward, cont_logit)."""
        hd = self._heads(obs, action)
        return (hd["d_proprio"], hd["d_goal_mean"], hd["voxel_logits"],
                hd["reward"], hd["cont_logit"])

    def predict(self, obs, action, sample=False):
        """Return (next_obs, reward, continuation) — used to roll imagination.

        next_obs has the same layout as a real observation: proprio/goal are
        obs + predicted delta, voxel is sigmoid(logits) in [0,1]. If `sample`
        and the model is probabilistic, the goal delta is drawn from its Gaussian
        (used by the visualizer to show per-member stochastic futures); the
        default (sample=False) uses the mean for a stable rollout.
        """
        hd = self._heads(obs, action)
        d_goal = hd["d_goal_mean"]
        if sample and hd["d_goal_logvar"] is not None:
            std = torch.exp(0.5 * hd["d_goal_logvar"])
            d_goal = d_goal + std * torch.randn_like(std)
        proprio = obs[:, :self._proprio_end] + hd["d_proprio"]
        goal    = obs[:, self._proprio_end:self._goal_end] + d_goal
        voxel   = torch.sigmoid(hd["voxel_logits"])
        next_obs = torch.cat([proprio, goal, voxel], dim=-1)
        return next_obs, hd["reward"], torch.sigmoid(hd["cont_logit"])

    # ── Training loss on a batch of real transitions ────────────────────────────
    def loss(self, obs, action, reward, next_obs, done):
        hd = self._heads(obs, action)

        tgt_proprio = next_obs[:, :self._proprio_end] - obs[:, :self._proprio_end]
        tgt_goal    = next_obs[:, self._proprio_end:self._goal_end] - obs[:, self._proprio_end:self._goal_end]
        tgt_voxel   = next_obs[:, self._goal_end:]
        cont_target = 1.0 - done

        l_proprio = F.mse_loss(hd["d_proprio"], tgt_proprio)
        if hd["d_goal_logvar"] is not None:
            # Gaussian negative log-likelihood (constant 0.5*log(2π) term dropped).
            s = hd["d_goal_logvar"]
            l_goal = 0.5 * (s + (tgt_goal - hd["d_goal_mean"]) ** 2 * torch.exp(-s)).mean()
        else:
            l_goal = F.mse_loss(hd["d_goal_mean"], tgt_goal)
        l_voxel   = F.binary_cross_entropy_with_logits(hd["voxel_logits"], tgt_voxel)
        l_reward  = F.mse_loss(hd["reward"], reward)
        l_cont    = F.binary_cross_entropy_with_logits(hd["cont_logit"], cont_target)

        total = (
            self.cfg.WM_W_PROPRIO * l_proprio
            + self.cfg.WM_W_GOAL  * l_goal
            + self.cfg.WM_W_VOXEL * l_voxel
            + self.cfg.WM_W_REWARD * l_reward
            + self.cfg.WM_W_CONT  * l_cont
        )
        metrics = {
            "wm_loss":         total.item(),
            "wm_proprio_loss": l_proprio.item(),
            "wm_goal_loss":    l_goal.item(),
            "wm_voxel_loss":   l_voxel.item(),
            "wm_reward_loss":  l_reward.item(),
            "wm_cont_loss":    l_cont.item(),
        }
        return total, metrics


class WorldModelEnsemble(nn.Module):
    """K independent WorldModel members → aleatoric/epistemic uncertainty.

    Being a single nn.Module (an nn.ModuleList of members), its state_dict()
    serializes all members, so Dreamer's existing checkpoint hook and a single
    optimizer over `.parameters()` work unchanged. Members diverge (nonzero
    epistemic uncertainty) via different random init + independent bootstrap
    minibatches (the sampling happens in Dreamer._train_world_model).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K   = cfg.ENSEMBLE_SIZE
        self.members = nn.ModuleList([WorldModel(cfg) for _ in range(self.K)])

        self._proprio_end = cfg.PROPRIOCEPTION_SIZE
        self._goal_end    = cfg.PROPRIOCEPTION_SIZE + cfg.GOAL_DELTA_SIZE
        # Which GOAL-stream dims carry the moving-target position (pig dx,dy,dz).
        self.unc_dims = list(getattr(cfg, "WM_UNCERTAINTY_DIMS", (0, 1, 2)))

    def predict(self, obs, action, sample=False):
        """Ensemble-mean (next_obs, reward, cont) — drop-in for WorldModel.predict."""
        nexts, rews, conts = self._all(obs, action, sample=sample)
        return nexts.mean(0), rews.mean(0), conts.mean(0)

    def predict_all(self, obs, action, sample=False):
        """Stacked per-member predictions: (K,B,D), (K,B), (K,B)."""
        return self._all(obs, action, sample=sample)

    def _all(self, obs, action, sample=False):
        nexts, rews, conts = [], [], []
        for m in self.members:
            n, r, c = m.predict(obs, action, sample=sample)
            nexts.append(n); rews.append(r); conts.append(c)
        return torch.stack(nexts), torch.stack(rews), torch.stack(conts)

    def predict_with_uncertainty(self, obs, action):
        """Ensemble-mean prediction plus (aleatoric, epistemic) on the target dims.

        Returns (next_obs_mean, reward_mean, cont_mean, aleatoric, epistemic),
        where aleatoric/epistemic are per-sample scalars (B,) summed over the
        pig-position dims:
            aleatoric = mean_k(exp(logvar_k))     (0 if members are deterministic)
            epistemic = var_k(mean_delta_k)
        """
        means, logvars, nexts, rews, conts = [], [], [], [], []
        for m in self.members:
            hd = m._heads(obs, action)
            gmean = hd["d_goal_mean"]
            means.append(gmean)
            if hd["d_goal_logvar"] is not None:
                logvars.append(hd["d_goal_logvar"])
            proprio = obs[:, :self._proprio_end] + hd["d_proprio"]
            goal    = obs[:, self._proprio_end:self._goal_end] + gmean
            voxel   = torch.sigmoid(hd["voxel_logits"])
            nexts.append(torch.cat([proprio, goal, voxel], dim=-1))
            rews.append(hd["reward"])
            conts.append(torch.sigmoid(hd["cont_logit"]))

        means = torch.stack(means)                       # (K, B, goal)
        mu    = means[:, :, self.unc_dims]               # (K, B, d)
        epistemic = mu.var(dim=0, unbiased=False).sum(dim=-1)   # (B,)
        if logvars:
            lv = torch.stack(logvars)[:, :, self.unc_dims]      # (K, B, d)
            aleatoric = torch.exp(lv).mean(dim=0).sum(dim=-1)   # (B,)
        else:
            aleatoric = torch.zeros_like(epistemic)

        next_obs = torch.stack(nexts).mean(0)
        reward   = torch.stack(rews).mean(0)
        cont     = torch.stack(conts).mean(0)
        return next_obs, reward, cont, aleatoric, epistemic


def _make_head_scalar(in_dim, hidden_dim):
    """Two-layer MLP ending in a single scalar (no final activation)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
