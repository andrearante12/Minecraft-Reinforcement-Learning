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

A stochastic recurrent (RSSM) variant is reserved for Phase 2 via the
WM_LATENT_DIM / USE_GRU config flags; this module implements the deterministic
path.
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
        self.goal_head    = nn.Linear(h, self.goal_size)      # residual delta
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

    def forward(self, obs, action):
        """Raw head outputs: (d_proprio, d_goal, voxel_logits, reward, cont_logit)."""
        feat = self._encode(obs)
        a    = self.act_embed(action)
        h    = self.trunk(torch.cat([feat, a], dim=-1))
        return (
            self.proprio_head(h),
            self.goal_head(h),
            self.voxel_head(h),
            self.reward_head(h).squeeze(-1),
            self.cont_head(h).squeeze(-1),
        )

    def predict(self, obs, action):
        """Return (next_obs, reward, continuation) — used to roll imagination.

        next_obs has the same layout as a real observation: proprio/goal are
        obs + predicted delta, voxel is sigmoid(logits) in [0,1].
        """
        d_proprio, d_goal, voxel_logits, reward, cont_logit = self.forward(obs, action)
        proprio = obs[:, :self._proprio_end] + d_proprio
        goal    = obs[:, self._proprio_end:self._goal_end] + d_goal
        voxel   = torch.sigmoid(voxel_logits)
        next_obs = torch.cat([proprio, goal, voxel], dim=-1)
        return next_obs, reward, torch.sigmoid(cont_logit)

    # ── Training loss on a batch of real transitions ────────────────────────────
    def loss(self, obs, action, reward, next_obs, done):
        d_proprio, d_goal, voxel_logits, reward_pred, cont_logit = self.forward(obs, action)

        tgt_proprio = next_obs[:, :self._proprio_end] - obs[:, :self._proprio_end]
        tgt_goal    = next_obs[:, self._proprio_end:self._goal_end] - obs[:, self._proprio_end:self._goal_end]
        tgt_voxel   = next_obs[:, self._goal_end:]
        cont_target = 1.0 - done

        l_proprio = F.mse_loss(d_proprio, tgt_proprio)
        l_goal    = F.mse_loss(d_goal, tgt_goal)
        l_voxel   = F.binary_cross_entropy_with_logits(voxel_logits, tgt_voxel)
        l_reward  = F.mse_loss(reward_pred, reward)
        l_cont    = F.binary_cross_entropy_with_logits(cont_logit, cont_target)

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


def _make_head_scalar(in_dim, hidden_dim):
    """Two-layer MLP ending in a single scalar (no final activation)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
