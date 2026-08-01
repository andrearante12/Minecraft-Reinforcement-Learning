"""
models/latent_actor_critic.py
-------------------------------
Actor-critic for the video world model (models/video_world_model.py). Operates
on RSSM features (deter ++ stoch, e.g. 256+32=288-dim), NOT on flat env
observations — this is the key difference from models/actor_critic.py.

NOT checkpoint-compatible with ActorCritic: its input space is the RSSM's
learned latent, not the 98-dim hunting observation vector, so warm-starting
to/from PPO/DQN/BC/dreamer checkpoints is not possible. algos/dreamer_video.py
checkpoints it under the standard "model_state" key (via BaseAgent.save), but
that state_dict is only loadable by another LatentActorCritic of the same
feat_dim.
"""

import math
import torch.nn as nn
from torch.distributions import Categorical


def _make_head(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LatentActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        feat_dim  = cfg.RSSM_DETER + cfg.WM_LATENT_DIM
        n_actions = cfg.N_ACTIONS
        head_h    = cfg.HEAD_HIDDEN

        self.actor  = _make_head(feat_dim, head_h, n_actions)
        self.critic = _make_head(feat_dim, head_h, 1)

        self._init_weights()

    def _init_weights(self):
        gain = math.sqrt(2.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, feat):
        """feat: (B, RSSM_DETER + WM_LATENT_DIM) -> (logits, value), same shapes as ActorCritic.forward."""
        return self.actor(feat), self.critic(feat)

    def get_distribution(self, feat):
        logits, _ = self.forward(feat)
        return Categorical(logits=logits)

    def get_value(self, feat):
        _, value = self.forward(feat)
        return value

    def evaluate_actions(self, feat, actions):
        logits, values = self.forward(feat)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy().mean()
