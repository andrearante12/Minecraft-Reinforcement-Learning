"""
models/vision_actor_critic.py
------------------------------
Model-free counterpart of the Dreamer agent for the vision hunting task
(--algo ppo_vision). It gets EXACTLY the inputs Dreamer's world model gets and
produces exactly the outputs Dreamer's policy produces:

    inputs   frozen DINOv2 tokens (T x D)  +  proprioception vector (11)
    outputs  action logits (N_ACTIONS)     +  state value

and uses the same building blocks, so the only thing that differs is what sits
in the middle:

                       Dreamer (algos/dreamer_video.py)     PPO (this file)
    token encoder      FeatureEncoder -> RSSM_EMBED         same module, same sizes
    proprio encoder    Linear+LN+ELU -> VWM_PROPRIO_EMBED   same
    internal model     recurrent RSSM (deter + stoch),      one feedforward Linear+LN+ELU
                       learned by reconstruction, acts      to the same feature width,
                       through imagined rollouts            no memory, no world model
    actor/critic       LatentActorCritic heads              same module, same sizes

Feature width: RSSM_DETER + WM_LATENT_DIM (288), so the heads are identical.

Not checkpoint-compatible with any other model (its input is the token tensor).
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.latent_actor_critic import LatentActorCritic
from models.video_world_model import FeatureEncoder
from models.frozen_encoder import token_shape


class VisionActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_tokens, token_dim = token_shape(cfg)
        feat_dim = cfg.RSSM_DETER + cfg.WM_LATENT_DIM

        # Same encoders as VideoWorldModel (feature mode).
        self.encoder = FeatureEncoder(n_tokens, token_dim, cfg.VIS_TOKEN_PROJ, cfg.RSSM_EMBED)
        self.vec_encoder = nn.Sequential(
            nn.Linear(cfg.INPUT_SIZE, cfg.VWM_PROPRIO_EMBED),
            nn.LayerNorm(cfg.VWM_PROPRIO_EMBED),
            nn.ELU(),
        )
        # The "internal model": a single feedforward fusion layer (no recurrence).
        self.fuse = nn.Sequential(
            nn.Linear(cfg.RSSM_EMBED + cfg.VWM_PROPRIO_EMBED, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ELU(),
        )
        # Same policy / value heads as Dreamer's actor-critic.
        self.heads = LatentActorCritic(cfg)
        # Same training-time privileged auxiliary head as the world model
        # (VWM_PRIV_AUX): predicts [sin b, cos b, dist, in_range] of the prey from
        # the fused feature. Labels only; the policy input is unchanged.
        self.priv_head = None
        if getattr(cfg, "VWM_PRIV_AUX", False):
            from models.video_world_model import _make_mlp_head
            self.priv_head = _make_mlp_head(feat_dim, cfg.WM_HEAD_HIDDEN, 4)

    def features(self, tokens, vec):
        return self.fuse(torch.cat([self.encoder(tokens), self.vec_encoder(vec)], dim=-1))

    def forward(self, tokens, vec):
        return self.heads(self.features(tokens, vec))

    def get_distribution(self, tokens, vec):
        logits, _ = self.forward(tokens, vec)
        return Categorical(logits=logits)

    def get_value(self, tokens, vec):
        return self.forward(tokens, vec)[1]

    def evaluate_actions(self, tokens, vec, actions):
        logits, values = self.forward(tokens, vec)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy().mean()
