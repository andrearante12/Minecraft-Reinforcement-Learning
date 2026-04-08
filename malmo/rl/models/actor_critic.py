"""
models/actor_critic.py
----------------------
Multi-stream Actor-Critic network.

Three input streams (proprioception, goal delta, voxel grid) are processed
independently, concatenated, then fed into separate actor and critic heads.
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _make_stream(in_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
    )


def _make_head(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        proprio_size = cfg.PROPRIOCEPTION_SIZE   # 6
        goal_size    = cfg.GOAL_DELTA_SIZE       # 3
        voxel_size   = cfg.GRID_SIZE             # 120
        n_actions    = cfg.N_ACTIONS             # 12

        proprio_h = cfg.PROPRIO_HIDDEN   # 64
        goal_h    = cfg.GOAL_HIDDEN      # 64
        voxel_h   = cfg.VOXEL_HIDDEN     # 128
        head_h    = cfg.HEAD_HIDDEN      # 256

        self.proprio_stream = _make_stream(proprio_size, proprio_h)
        self.goal_stream    = _make_stream(goal_size, goal_h)
        self.voxel_stream   = _make_stream(voxel_size, voxel_h)

        merge_dim = proprio_h + goal_h + voxel_h  # 256

        self.actor  = _make_head(merge_dim, head_h, n_actions)
        self.critic = _make_head(merge_dim, head_h, 1)

        # Store split indices for config-driven observation slicing
        self._proprio_size = proprio_size
        self._goal_end     = proprio_size + goal_size

        self._init_weights()

    def _init_weights(self):
        gain = math.sqrt(2.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        # Override final layers with specific gains
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, obs):
        proprio = obs[:, :self._proprio_size]
        goal    = obs[:, self._proprio_size:self._goal_end]
        voxel   = obs[:, self._goal_end:]

        p = self.proprio_stream(proprio)
        g = self.goal_stream(goal)
        v = self.voxel_stream(voxel)

        merged = torch.cat([p, g, v], dim=-1)
        return self.actor(merged), self.critic(merged)

    def get_distribution(self, obs):
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

    def evaluate_actions(self, obs, actions):
        """Returns log_probs, values, entropy for a batch — used in PPO update."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy().mean()
