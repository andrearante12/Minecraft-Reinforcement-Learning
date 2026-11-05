"""
algos/ppo_vision.py
--------------------
PPO on the same observations Dreamer uses in the vision hunting task: a camera
frame (encoded here by the same frozen DINOv2) plus the 11 proprio values. A
model-free baseline whose only difference from algos/dreamer_video.py is the
internal model (models/vision_actor_critic.py: one feedforward layer instead of
a recurrent world model + imagination).

Everything else is inherited from algos/ppo.py unchanged: GAE, clipped policy and
value losses, advantage/reward normalisation, LR + entropy schedules. The
proprio vector is running-mean/std normalised as in PPO; the DINO tokens are
fed as-is (LayerNorm inside the encoder), exactly as Dreamer feeds them.

Obs are (vec, frame) tuples (EnvClient(video=True)), so, like DreamerVideo, this
agent overrides collect_steps. Tokens for the rollout are kept in a side array
next to the inherited RolloutBuffer (fp16).
"""

import numpy as np
import torch

from algos.ppo import PPO
from models.frozen_encoder import build_frozen_encoder


class PPOVision(PPO):
    def __init__(self, model, cfg, n_envs=1):
        super().__init__(model, cfg, n_envs=n_envs)
        self.frozen_encoder = build_frozen_encoder(cfg, self.device)
        T, D = self.frozen_encoder.n_tokens, self.frozen_encoder.dim
        b = self.buffer
        self._tok = np.zeros((b.n_steps_per_env, n_envs, T, D), dtype=np.float16)
        self._tok_t = None
        self.steps_done = 0
        # Privileged aux labels (same as DreamerVideo): the 12-dim target stream
        # for the obs at each step arrives in the PREVIOUS step's info.
        self.priv_aux = getattr(cfg, "VWM_PRIV_AUX", False) and getattr(model, "priv_head", None) is not None
        self._priv = np.zeros((b.n_steps_per_env, n_envs, 12), dtype=np.float32)
        self._priv_t = None
        self._env_priv = [None] * n_envs
        print("PPOVision: frozen encoder {0} | tokens {1}x{2} | params {3}".format(
            cfg.VIS_ENCODER, T, D, sum(p.numel() for p in self.model.parameters())))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _encode(self, obs_list):
        """[(vec, frame), ...] -> (tokens (N,T,D) float tensor on device, raw vecs (N,11))."""
        vecs = np.stack([np.asarray(o[0], dtype=np.float32) for o in obs_list])
        frames = np.stack([np.asarray(o[1], dtype=np.uint8) for o in obs_list])
        with torch.no_grad():
            tokens = self.frozen_encoder(frames)
        return tokens, vecs

    def _norm_vec(self, vecs):
        return self.obs_rms.normalize(vecs, self.cfg.NORM_CLIP) if self.obs_rms else vecs

    # ── data collection ──────────────────────────────────────────────────────
    def collect_steps(self, envs, obs_all):
        n_envs = len(envs)
        tokens, vecs = self._encode(obs_all)
        if self.obs_rms:
            self.obs_rms.update(vecs)
        vec_n = self._norm_vec(vecs)
        vec_t = torch.as_tensor(vec_n, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            tok_f = tokens.float()
            dist = self.model.get_distribution(tok_f, vec_t)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = self.model.get_value(tok_f, vec_t).squeeze(-1)
        actions_np = actions.cpu().numpy()

        next_obs_all = [None] * n_envs
        rewards = np.zeros(n_envs, dtype=np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        infos = [None] * n_envs
        for i, env in enumerate(envs):
            next_obs_all[i], rewards[i], d, infos[i] = env.step(int(actions_np[i]))
            dones[i] = float(d)
        self.steps_done += n_envs

        if self.reward_rms:
            self.reward_rms.update(rewards)
            r_store = np.clip(rewards / np.sqrt(self.reward_rms.var + 1e-8),
                              -self.cfg.NORM_CLIP, self.cfg.NORM_CLIP)
        else:
            r_store = rewards

        self._tok[self.buffer.ptr] = tokens.cpu().numpy().astype(np.float16)
        if self.priv_aux:
            for i in range(n_envs):
                p = self._env_priv[i]
                self._priv[self.buffer.ptr, i] = 0.0 if p is None else np.asarray(p, np.float32)
                self._env_priv[i] = None if dones[i] else infos[i].get("privileged")
        self.buffer.add(vec_n, actions_np, r_store, dones,
                        log_probs.cpu().numpy(), values.cpu().numpy())
        return next_obs_all, rewards, dones, infos

    def collect_step(self, env, obs):
        nxt, r, d, info = self.collect_steps([env], [obs])
        return nxt[0], r[0], bool(d[0]), info[0]

    # ── PPO hooks ────────────────────────────────────────────────────────────
    def _bootstrap_values(self, last_obs):
        tokens, vecs = self._encode(list(last_obs))
        vec_t = torch.as_tensor(self._norm_vec(vecs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model.get_value(tokens.float(), vec_t).squeeze(-1)

    def _evaluate(self, obs_t, actions_t, idx):
        tok = self._tok_t[idx].float()
        return self.model.evaluate_actions(tok, obs_t[idx], actions_t[idx])

    def _aux_loss(self, obs_t, idx):
        """Privileged aux loss on this minibatch (0 when the head is off): MSE of
        priv_head(features) vs [sin b, cos b, dist, in_range] on sensed steps."""
        if not self.priv_aux:
            return None
        from models.video_world_model import privileged_targets
        targets, sensed = privileged_targets(self._priv_t[idx])
        if sensed.sum() < 1:
            return None
        feat = self.model.features(self._tok_t[idx].float(), obs_t[idx])
        err = ((self.model.priv_head(feat) - targets) ** 2).sum(-1)
        return (err * sensed).sum() / sensed.sum()

    def update(self, last_obs):
        total = self.buffer.total_steps
        T, D = self._tok.shape[-2:]
        self._tok_t = torch.as_tensor(self._tok.reshape(total, T, D), device=self.device)
        self._priv_t = torch.as_tensor(self._priv.reshape(total, 12), device=self.device)
        logs = super().update(last_obs)
        self._tok_t = None
        self._priv_t = None
        return logs

    # ── demo warm start (behaviour cloning) ─────────────────────────────────
    def pretrain_from_demos(self, path, n_steps, batch_size=256, log_every=250):
        """Warm-start the actor on a frames demo directory (utils/demo_io.py): the
        same demos Dreamer's --demo-path loads. Cross-entropy of the policy on the
        demonstrator's actions, from the same frozen DINOv2 tokens + proprio the
        agent sees. The proprio normaliser is seeded from the demo vectors first,
        so the cloned policy sees the same input scale it will see online. Only
        successful episodes are used when cfg.DEMO_ONLY_SUCCESS (as Dreamer does)."""
        import torch.nn.functional as F
        from utils.demo_io import is_frame_demo_dir, load_index, load_episode, episode_features
        if not is_frame_demo_dir(path):
            print("pretrain_from_demos: {0} is not a frames demo directory — skipping".format(path))
            return
        only_success = getattr(self.cfg, "DEMO_ONLY_SUCCESS", False)
        thumb = getattr(self.cfg, "VIS_THUMB", 64)
        toks, vecs, acts = [], [], []
        for entry in load_index(path)["episodes"]:
            if only_success and entry.get("outcome") != "killed":
                continue
            ep = load_episode(path, entry, frames=False)
            feats, _ = episode_features(path, entry, self.frozen_encoder, thumb)
            toks.append(feats); vecs.append(ep["vecs"].astype(np.float32)); acts.append(ep["actions"])
        if not toks:
            print("pretrain_from_demos: no usable demo episodes in {0}".format(path))
            return
        toks, vecs, acts = np.concatenate(toks), np.concatenate(vecs), np.concatenate(acts).astype(np.int64)
        if self.obs_rms:
            self.obs_rms.update(vecs)
        tok_t = torch.as_tensor(toks, device=self.device)                      # fp16 on device
        vec_t = torch.as_tensor(self._norm_vec(vecs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(acts, device=self.device)
        n = len(acts)
        print("Behaviour-cloning warm start: {0} demo steps, {1} updates (batch {2})".format(
            n, n_steps, batch_size))
        self.model.train()
        for i in range(1, n_steps + 1):
            idx = torch.randint(0, n, (min(batch_size, n),), device=self.device)
            logits, _ = self.model(tok_t[idx].float(), vec_t[idx])
            loss = F.cross_entropy(logits, act_t[idx])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
            self.optimizer.step()
            if i % log_every == 0 or i == n_steps:
                with torch.no_grad():
                    acc = (logits.argmax(-1) == act_t[idx]).float().mean().item()
                print("  [bc {0:>5}/{1}] loss {2:.4f} | match teacher {3:.0%}".format(i, n_steps, loss.item(), acc))

    # ── inference ────────────────────────────────────────────────────────────
    def select_action(self, obs, greedy=False):
        tokens, vecs = self._encode([obs])
        vec_t = torch.as_tensor(self._norm_vec(vecs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(tokens.float(), vec_t)
            if greedy:
                return int(logits.argmax(dim=-1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())
