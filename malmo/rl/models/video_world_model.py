"""
models/video_world_model.py
----------------------------
Video-based world model (DreamerV1-style RSSM) — a SEPARATE architecture from
models/world_model.py, used only by algos/dreamer_video.py (--algo dreamer_video).

Where the vector WorldModel predicts the next observation directly in
observation space (no decoder, since obs are already low-dim state vectors),
this model learns a compact RECURRENT LATENT from raw frames:

    frame, vec  --[ConvEncoder, vec MLP]-->  embed
    (deter, stoch), action, embed  --[RSSM]-->  (deter', stoch')
    feat = concat(deter', stoch')  --[ConvDecoder + heads]-->  frame, vec, reward, continuation

The RSSM (Hafner et al., "Dream to Control", 2019) carries a GRU deterministic
state plus a Gaussian stochastic latent. The stochastic latent is what lets the
model represent un-actioned randomness in the world (e.g. a fleeing animal's
motion) that a purely deterministic recurrence cannot. Training fits a
posterior (uses the real frame) and a prior (predicts blind, action-only) at
every step, and pushes them together with a KL term (floored at WM_FREE_NATS
"free nats" so the posterior isn't forced to collapse onto an under-trained
prior early in training). Imagination later rolls out PRIOR-ONLY from a real
posterior start state, since the whole point is not needing real frames.

This module has no coupling to models/world_model.py or models/actor_critic.py
— the two world-model architectures are independent by design so neither can
regress the other. See models/latent_actor_critic.py for the paired policy
(acts on RSSM features, NOT on flat obs — not checkpoint-compatible with
ActorCritic/PPO/DQN/BC/dreamer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp_head(in_dim, hidden_dim, out_dim):
    """Two-layer MLP ending in `out_dim` (no final activation)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _stack_states(states):
    """List of L per-step state dicts (each value (B,*)) -> dict of (B,L,*)."""
    keys = states[0].keys()
    return {k: torch.stack([s[k] for s in states], dim=1) for k in keys}


class ConvEncoder(nn.Module):
    """64x64xC frame(s) -> flat embedding. Standard 4-conv DreamerV1 encoder.

    Accepts (B,C,H,W) or (B,L,C,H,W); returns (B,embed) or (B,L,embed).
    With depth=32 and a 64x64 input this yields exactly a 1024-dim embedding
    (channels 32->64->128->256, spatial 64->31->14->6->2, 256*2*2=1024).
    """

    def __init__(self, in_channels, depth):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, depth, 4, stride=2), nn.ELU(),
            nn.Conv2d(depth, depth * 2, 4, stride=2), nn.ELU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2), nn.ELU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2), nn.ELU(),
        )

    def forward(self, x):
        squeeze_l = (x.dim() == 4)
        if squeeze_l:
            x = x.unsqueeze(1)
        B, L, C, H, W = x.shape
        x = self.net(x.reshape(B * L, C, H, W))
        x = x.reshape(B, L, -1)
        return x.squeeze(1) if squeeze_l else x


class ConvDecoder(nn.Module):
    """RSSM feature -> reconstructed 64x64xC frame. Mirrors ConvEncoder.

    Accepts (B,feat) or (B,L,feat); returns (B,C,H,W) or (B,L,C,H,W) — a raw
    pixel-space mean (no final activation; trained with MSE against frames
    normalized to roughly [-0.5, 0.5]).
    """

    def __init__(self, feat_dim, out_channels, depth, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(feat_dim, embed_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, depth * 4, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2), nn.ELU(),
            nn.ConvTranspose2d(depth, out_channels, 6, stride=2),
        )

    def forward(self, feat):
        squeeze_l = (feat.dim() == 2)
        if squeeze_l:
            feat = feat.unsqueeze(1)
        B, L, D = feat.shape
        x = self.fc(feat.reshape(B * L, D)).reshape(B * L, self.embed_dim, 1, 1)
        x = self.net(x)
        _, C, H, W = x.shape
        x = x.reshape(B, L, C, H, W)
        return x.squeeze(1) if squeeze_l else x


class RSSM(nn.Module):
    """Recurrent State-Space Model: GRU deterministic state + Gaussian stochastic latent.

    State dict keys: "deter" (B,deter_dim), "stoch" (B,stoch_dim),
    "mean"/"std" (B,stoch_dim) — the Gaussian the stoch sample was drawn from
    (needed for the KL loss; prior and posterior each carry their own).
    """

    def __init__(self, embed_dim, n_actions, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.n_actions = n_actions

        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_dim + n_actions, hidden_dim), nn.ELU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def initial(self, batch, device):
        return {
            "deter": torch.zeros(batch, self.deter_dim, device=device),
            "stoch": torch.zeros(batch, self.stoch_dim, device=device),
            "mean":  torch.zeros(batch, self.stoch_dim, device=device),
            "std":   torch.ones(batch, self.stoch_dim, device=device),
        }

    def _sample(self, stats):
        mean, std_raw = stats.chunk(2, dim=-1)
        std = F.softplus(std_raw) + 0.1
        stoch = mean + std * torch.randn_like(mean)
        return mean, std, stoch

    def img_step(self, prev_state, prev_action):
        """Prior transition: blind to the current frame, action-conditioned only."""
        x = self.pre_gru(torch.cat([prev_state["stoch"], prev_action], dim=-1))
        deter = self.gru(x, prev_state["deter"])
        mean, std, stoch = self._sample(self.prior_net(deter))
        return {"deter": deter, "stoch": stoch, "mean": mean, "std": std}

    def obs_step(self, prev_state, prev_action, embed):
        """Posterior transition: prior + the current frame/vec embedding."""
        prior = self.img_step(prev_state, prev_action)
        mean, std, stoch = self._sample(self.post_net(torch.cat([prior["deter"], embed], dim=-1)))
        post = {"deter": prior["deter"], "stoch": stoch, "mean": mean, "std": std}
        return post, prior

    def observe(self, embeds, actions):
        """Posterior/prior over a full sequence.

        embeds:  (B,L,E) — encoder output per step.
        actions: (B,L,A) one-hot — the action taken to REACH each step (i.e.
                 already shifted by the caller; action at t=0 is all-zero).
        Each sampled window is treated as starting from the RSSM's zero
        initial state (matches DreamerV1's chunked-training convention);
        SequenceReplayBuffer guarantees a window never crosses an episode
        boundary, so this never straddles a real episode reset mid-window.
        Returns (posts, priors), each a dict of (B,L,*) tensors.
        """
        B, L, _ = embeds.shape
        state = self.initial(B, embeds.device)
        posts, priors = [], []
        for t in range(L):
            post, prior = self.obs_step(state, actions[:, t], embeds[:, t])
            posts.append(post)
            priors.append(prior)
            state = post
        return _stack_states(posts), _stack_states(priors)

    @staticmethod
    def get_feat(state):
        return torch.cat([state["stoch"], state["deter"]], dim=-1)

    def kl_loss(self, post, prior):
        """KL(post || prior), summed over the stochastic dims -> (B,L) (or (B,) for single-step states)."""
        post_dist  = torch.distributions.Normal(post["mean"], post["std"])
        prior_dist = torch.distributions.Normal(prior["mean"], prior["std"])
        return torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1)


class VideoWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_actions  = cfg.N_ACTIONS
        self.deter_dim  = cfg.RSSM_DETER
        self.stoch_dim  = cfg.WM_LATENT_DIM
        self.free_nats  = getattr(cfg, "WM_FREE_NATS", 1.0)
        self.kl_scale   = getattr(cfg, "WM_KL_SCALE", 1.0)
        self.vec_size   = cfg.INPUT_SIZE

        depth = cfg.VWM_CNN_DEPTH
        self.encoder = ConvEncoder(cfg.VIDEO_CHANNELS, depth)
        self.vec_encoder = nn.Sequential(
            nn.Linear(self.vec_size, cfg.VWM_PROPRIO_EMBED),
            nn.LayerNorm(cfg.VWM_PROPRIO_EMBED),
            nn.ELU(),
        )
        embed_dim = cfg.RSSM_EMBED + cfg.VWM_PROPRIO_EMBED
        self.rssm = RSSM(embed_dim, self.n_actions, self.deter_dim, self.stoch_dim, cfg.RSSM_HIDDEN)

        feat_dim = self.deter_dim + self.stoch_dim
        self.decoder     = ConvDecoder(feat_dim, cfg.VIDEO_CHANNELS, depth, cfg.RSSM_EMBED)
        self.vec_head    = _make_mlp_head(feat_dim, cfg.WM_HEAD_HIDDEN, self.vec_size)
        # Reward/continuation are conditioned on (feat, action) explicitly — same
        # convention as models/world_model.py's (obs, action) -> reward — rather
        # than implicitly via the next latent state. This keeps the training
        # target alignment exact: SequenceReplayBuffer stores reward[t] as the
        # reward for taking action[t] FROM the state observed at t, so the head
        # must see action[t] directly instead of relying on it being folded into
        # a (shifted) next state.
        self.reward_head = _make_mlp_head(feat_dim + self.n_actions, cfg.WM_HEAD_HIDDEN, 1)
        self.cont_head   = _make_mlp_head(feat_dim + self.n_actions, cfg.WM_HEAD_HIDDEN, 1)

    # ── Encoding ─────────────────────────────────────────────────────────────
    def _embed(self, frames, vecs):
        """frames: (B,L,H,W,C) uint8 or float or (B,H,W,C); vecs matching leading dims."""
        frames_norm = frames.float() / 255.0 - 0.5
        if frames_norm.dim() == 4:
            frames_chw = frames_norm.permute(0, 3, 1, 2)
        else:
            frames_chw = frames_norm.permute(0, 1, 4, 2, 3)
        img_embed = self.encoder(frames_chw)
        vec_embed = self.vec_encoder(vecs)
        return torch.cat([img_embed, vec_embed], dim=-1), frames_norm

    @staticmethod
    def _shift_actions(actions_onehot):
        """Action at t=0 is all-zero (nothing was taken to reach the first step)."""
        prev = torch.zeros_like(actions_onehot)
        prev[:, 1:] = actions_onehot[:, :-1]
        return prev

    def decode(self, feat):
        """RSSM feature -> (frame_mean [-0.5,0.5]-space, vec_mean). Used by the viz."""
        return self.decoder(feat), self.vec_head(feat)

    def predict_reward_cont(self, feat, action_onehot):
        """(feat, action taken FROM this state) -> (reward, continuation). Used
        both by loss() and by the imagination rollout in algos/dreamer_video.py."""
        feat_act = torch.cat([feat, action_onehot], dim=-1)
        reward = self.reward_head(feat_act).squeeze(-1)
        cont = torch.sigmoid(self.cont_head(feat_act).squeeze(-1))
        return reward, cont

    @torch.no_grad()
    def encode_step(self, state, prev_action_onehot, frame, vec):
        """Online single-step posterior update for acting (no sequence dim)."""
        embed, _ = self._embed(frame, vec)
        post, _ = self.rssm.obs_step(state, prev_action_onehot, embed)
        return post

    # ── Training loss on a batch of real (masked) sequences ────────────────────
    def loss(self, frames, vecs, actions, rewards, dones, mask):
        """
        frames: (B,L,H,W,C) uint8   vecs: (B,L,D) f32   actions: (B,L) int64
        rewards/dones/mask: (B,L) f32 — mask=1 for real steps, 0 for padding.
        Returns (total, metrics_dict, posts) — posts is DETACHED (safe to reuse
        as imagination start states after backward()).
        """
        eps = 1e-8
        actions_onehot = F.one_hot(actions, self.n_actions).float()
        prev_actions = self._shift_actions(actions_onehot)

        embed, frames_norm = self._embed(frames, vecs)
        posts, priors = self.rssm.observe(embed, prev_actions)
        feat = self.rssm.get_feat(posts)                      # (B,L,feat_dim)

        recon_frame, recon_vec = self.decode(feat)
        feat_act = torch.cat([feat, actions_onehot], dim=-1)  # unshifted: action taken FROM this state
        reward_pred = self.reward_head(feat_act).squeeze(-1)
        cont_logit  = self.cont_head(feat_act).squeeze(-1)

        frames_chw = frames_norm.permute(0, 1, 4, 2, 3)
        mask_sum = mask.sum().clamp_min(eps)

        n_pix = frames_chw.shape[2] * frames_chw.shape[3] * frames_chw.shape[4]
        l_image = (((recon_frame - frames_chw) ** 2).sum(dim=(2, 3, 4)) * mask).sum() / (mask_sum * n_pix)
        l_vec   = (((recon_vec - vecs) ** 2).sum(dim=-1) * mask).sum() / (mask_sum * self.vec_size)
        l_reward = (((reward_pred - rewards) ** 2) * mask).sum() / mask_sum
        cont_target = 1.0 - dones
        l_cont = (F.binary_cross_entropy_with_logits(cont_logit, cont_target, reduction="none") * mask).sum() / mask_sum

        kl_raw = self.rssm.kl_loss(posts, priors)              # (B,L)
        l_kl = (torch.clamp(kl_raw, min=self.free_nats) * mask).sum() / mask_sum

        total = (
            self.cfg.VWM_W_IMAGE   * l_image
            + self.cfg.VWM_W_PROPRIO * l_vec
            + self.cfg.VWM_W_REWARD  * l_reward
            + self.cfg.VWM_W_CONT    * l_cont
            + self.kl_scale          * l_kl
        )
        metrics = {
            "vwm_loss":    total.item(),
            "vwm_image":   l_image.item(),
            "vwm_proprio": l_vec.item(),
            "vwm_reward":  l_reward.item(),
            "vwm_cont":    l_cont.item(),
            "vwm_kl":      l_kl.item(),
        }
        posts_detached = {k: v.detach() for k, v in posts.items()}
        return total, metrics, posts_detached
