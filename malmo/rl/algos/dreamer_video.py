"""
algos/dreamer_video.py
------------------------
Video-based model-based RL agent — implements BaseAgent. A SEPARATE algo from
algos/dreamer.py (not a subclass): observation type, acting statefulness,
replay buffer, and imagination substrate all differ enough that sharing code
would force branches into the frozen vector-Dreamer path.

Idea (same as dreamer.py, in pixel space): collect real (frame, vec, action,
reward, done) transitions from a video-enabled env (VIDEO_ENABLED=True), fit a
recurrent latent world model (models/video_world_model.py, a DreamerV1-style
RSSM) to them, then train a latent actor-critic almost entirely inside the
model's imagination — a PRIOR-ONLY rollout from real posterior start states.

Acting is STATEFUL: unlike the vector agent (which acts directly on a flat
obs), this agent must carry an RSSM posterior state across steps within an
episode (the latent is a summary of everything seen so far in the episode,
not just the current frame). `collect_step`/`collect_steps` track one RSSM
state + previous action per parallel env, resetting on `done`.

Frozen visual encoder (cfg.VWM_INPUT == "features", e.g. hunting_vision):
frames are encoded by a frozen pretrained backbone (models/frozen_encoder.py,
DINOv2) here in the trainer, and the replay buffer stores those tokens (+ a
64x64 thumbnail for the viewers) instead of pixels.

Demos (DreamerFD, ported from dreamer.py; cfg.DEMO_PATH = a utils/demo_io.py
frames directory): a second sequence buffer supplies DEMO_WM_FRACTION of each
world-model minibatch, DEMO_START_FRACTION of imagination start states, and a
decaying behaviour-cloning anchor on the demo posteriors (DEMO_BC_COEF).
pretrain_world_model() fits the world model on demos before any live step.

Checkpoint compatibility: NONE with dreamer.py / PPO / DQN / BC. self.model is
a LatentActorCritic (acts on RSSM features, not flat obs) and the extra state
key is "video_world_model_state" (deliberately different from dreamer.py's
"world_model_state") so a wrong-algo checkpoint load fails loudly on a shape
mismatch rather than silently loading the wrong architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from algos.base_agent import BaseAgent
from algos.sequence_replay_buffer import SequenceReplayBuffer
from models.video_world_model import VideoWorldModel


class DreamerVideo(BaseAgent):
    def __init__(self, model, cfg, n_envs=1):
        self.model  = model             # LatentActorCritic (policy + value on RSSM features)
        self.cfg    = cfg
        self.n_envs = n_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR, eps=1e-5)

        self.world_model  = VideoWorldModel(cfg).to(self.device)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=cfg.VWM_LR, eps=1e-5)
        self.n_actions     = cfg.N_ACTIONS

        self.buffer = SequenceReplayBuffer(cfg.SEQ_BUFFER_CAPACITY, cfg.WM_SEQ_LEN, cfg.VWM_MIN_SEQ)
        self.steps_done = 0
        self._real_reward_ema = 0.0     # EMA (1/2000) of collected per-step reward

        # Frozen pretrained encoder (feature mode only; never trained or saved).
        self.feature_mode   = (getattr(cfg, "VWM_INPUT", "pixels") == "features")
        self.obs_key        = "feats" if self.feature_mode else "frames"
        self.frozen_encoder = None
        if self.feature_mode:
            from models.frozen_encoder import build_frozen_encoder
            self.frozen_encoder = build_frozen_encoder(cfg, self.device)
            self.thumb_size = getattr(cfg, "VIS_THUMB", 64)

        # Training-time privileged aux labels (see VideoWorldModel.priv_head).
        self.priv_aux = getattr(cfg, "VWM_PRIV_AUX", False)

        # DreamerFD-style demos (cfg.DEMO_PATH = frames demo directory).
        self.demo_buffer   = None
        self._demo_bc_coef = getattr(cfg, "DEMO_BC_COEF", 0.0)
        demo_path = getattr(cfg, "DEMO_PATH", None)
        if demo_path:
            self._load_demos(demo_path)
        # Hazard episodes (scripted walks into pits / lava): HAZARD_WM_FRACTION of
        # each world-model batch, so the reward and continuation heads learn that
        # hazards END the episode with REWARD_PIT / REWARD_LAVA. Never used for
        # behaviour cloning (rows sit before the demo rows, which BC reads last).
        self.hazard_buffer = None
        hz = getattr(cfg, "HAZARD_DEMO_PATH", None)
        if hz:
            self.hazard_buffer = self._load_demos(hz, outcomes=("stuck_in_pit", "lava", "fell"),
                                                  assign=False)

        # Per-env RSSM posterior state + previous action (for stateful acting).
        self._env_state    = [self._initial_state() for _ in range(n_envs)]
        self._env_prev_act = [self._zero_action() for _ in range(n_envs)]
        # Privileged target stream for the CURRENT obs of each env (arrives in
        # the previous step's info; unknown right after reset -> zeros = unsensed).
        self._env_priv = [None] * n_envs

        # Separate persistent state for select_action() (e.g. evaluate.py callers).
        # LIMITATION: assumes sequential single-episode calls; callers must call
        # reset_eval_state() between evaluation episodes.
        self._eval_state    = None
        self._eval_prev_act = None

        self.initial_entropy_coef = cfg.IMAG_ENTROPY_COEF
        self.current_entropy_coef = cfg.IMAG_ENTROPY_COEF
        self.initial_lr           = cfg.LR
        self.current_lr           = cfg.LR

        print("DreamerVideo initialized on device:", self.device)
        print("  Video world model (RSSM) params:", sum(p.numel() for p in self.world_model.parameters()))
        print("  Latent actor-critic params:", sum(p.numel() for p in self.model.parameters()))
        print("  RSSM: deter={0} stoch={1} | seq_len={2} | video {3}x{4}x{5}".format(
            cfg.RSSM_DETER, cfg.WM_LATENT_DIM, cfg.WM_SEQ_LEN,
            cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS))
        print("  Imagination horizon: {0} | batch: {1} | lambda: {2}".format(
            cfg.IMAG_HORIZON, cfg.IMAG_BATCH, cfg.IMAG_LAMBDA))
        print("  Learning starts at {0} real transitions".format(cfg.LEARNING_STARTS))
        if self.feature_mode:
            print("  Frozen encoder: {0} | tokens {1}x{2} | input {3}px".format(
                cfg.VIS_ENCODER, self.frozen_encoder.n_tokens, self.frozen_encoder.dim,
                self.frozen_encoder.res))

    # ── Stateful acting helpers ──────────────────────────────────────────────
    def _initial_state(self):
        return self.world_model.rssm.initial(1, self.device)

    def _zero_action(self):
        return torch.zeros(1, self.n_actions, device=self.device)

    def _act_from_state(self, state, greedy):
        feat = self.world_model.rssm.get_feat(state)
        with torch.no_grad():
            logits, _ = self.model(feat)
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        return int(action.item()), F.one_hot(action, self.n_actions).float()

    @torch.no_grad()
    def _plan_action(self, state):
        """Lookahead planner (evaluation only; cfg.PLAN_AT_ACT).

        The pig's movement is uncertain, so each candidate action is judged over
        several imagined futures, each sampling a different stochastic latent
        (i.e. a different reaction), and the candidate with the best average
        (or worst-case) score is played. A future's score is its discounted
        predicted reward over PLAN_HORIZON steps plus the critic's value at the
        end. Reuses the same RSSM prior / reward / continuation pieces as
        _imagine_and_train. Returns (action, one-hot, per-candidate scores dict).
        """
        cfg   = self.cfg
        rssm  = self.world_model.rssm
        C     = min(int(getattr(cfg, "PLAN_CANDIDATES", 4)), self.n_actions)
        K     = max(1, int(getattr(cfg, "PLAN_SAMPLES", 8)))
        H     = max(1, int(getattr(cfg, "PLAN_HORIZON", 8)))
        gamma = cfg.GAMMA

        logits, _ = self.model(rssm.get_feat(state))
        greedy = int(logits[0].argmax().item())
        # PLAN_ACTION_SET: the planner only re-decides among these actions (aim /
        # attack by default) and only when the actor itself would pick one of
        # them; locomotion stays with the actor, which the BC anchor keeps away
        # from hazards the world model has rarely seen end an episode.
        allowed = self._plan_allowed()
        if allowed is not None and greedy not in allowed:
            self.last_plan = {}
            return greedy, F.one_hot(torch.tensor([greedy], device=self.device), self.n_actions).float(), {}
        if allowed is not None:
            masked = logits[0].clone()
            masked[[a for a in range(self.n_actions) if a not in allowed]] = float("-inf")
            C = min(C, len(allowed))
            cands = masked.topk(C).indices
        else:
            cands = logits[0].topk(C).indices                   # (C,) candidate actions
        first = cands.repeat_interleave(K)                      # (C*K,)
        st = {k: v.expand(C * K, -1).contiguous() for k, v in state.items()}

        ret  = torch.zeros(C * K, device=self.device)
        disc = torch.ones(C * K, device=self.device)
        alive = torch.ones(C * K, device=self.device)          # product of continuation probs
        a = first
        for t in range(H):
            feat = rssm.get_feat(st)
            if t > 0:                                           # actor continues the plan
                a = Categorical(logits=self.model(feat)[0]).sample()
            a_onehot = F.one_hot(a, self.n_actions).float()
            # samples z: a different reaction per future; pessimistic reward if an ensemble is on
            r, cont, st = self.world_model.step_reward_cont(
                st, a_onehot, pessimism=float(getattr(cfg, "IMAG_REWARD_PESSIMISM", 0.0)))
            ret  = ret + disc * r
            disc = disc * gamma * cont
            alive = alive * cont
        _, last_v = self.model(rssm.get_feat(st))
        ret = (ret + disc * last_v.squeeze(-1)).view(C, K)
        p_end = (1.0 - alive).view(C, K).mean(dim=1)            # P(episode ends within H) per candidate

        if getattr(cfg, "PLAN_SCORE", "mean") == "worst":
            scores = ret.min(dim=1).values
        else:
            scores = ret.mean(dim=1)
        self.last_plan = {int(c): float(s) for c, s in zip(cands, scores)}
        # Override the actor only with a reason: PLAN_MARGIN = how much the best
        # candidate must beat the actor's own choice by (score units); PLAN_VETO_P:
        # with PLAN_MODE="veto" the planner only acts when the actor's choice ends
        # the episode (hazard) with at least this probability, then takes the
        # candidate with the lowest P(end).
        mode = getattr(cfg, "PLAN_MODE", "argmax")
        cl = cands.tolist()
        if greedy in cl:
            gi = cl.index(greedy)
            if mode == "veto":
                if float(p_end[gi]) >= float(getattr(cfg, "PLAN_VETO_P", 0.5)):
                    best = cands[p_end.argmin()]
                else:
                    best = cands[gi]
            else:
                margin = float(getattr(cfg, "PLAN_MARGIN", 0.0))
                bi = int(scores.argmax())
                best = cands[bi] if float(scores[bi]) > float(scores[gi]) + margin else cands[gi]
        else:
            best = cands[scores.argmax()]
        return int(best.item()), F.one_hot(best.view(1), self.n_actions).float(), self.last_plan

    def _plan_allowed(self):
        names = getattr(self.cfg, "PLAN_ACTION_SET", None)
        if not names:
            return None
        if getattr(self, "_plan_allowed_cache", None) is None:
            all_names = [a[0] for a in self.cfg.ACTIONS]
            self._plan_allowed_cache = set(all_names.index(n) for n in names if n in all_names)
        return self._plan_allowed_cache

    # ── Observation encoding ─────────────────────────────────────────────────
    def _encode_frames(self, frames):
        """List of (H,W,C) uint8 frames -> (world-model input tensor (N,...) on
        device, per-frame dicts to store in the replay buffer)."""
        frames = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
        if self.feature_mode:
            from models.frozen_encoder import make_thumbs
            feats = self.frozen_encoder(frames)
            feats_np = feats.cpu().numpy()
            thumbs = make_thumbs(frames, self.thumb_size)
            return feats, [{"feats": feats_np[i], "thumbs": thumbs[i]} for i in range(len(frames))]
        return (torch.as_tensor(frames, device=self.device),
                [{"frames": frames[i]} for i in range(len(frames))])

    def model_input(self, frame):
        """One frame -> the (1,...) tensor encode_step expects (used by the viewers)."""
        return self._encode_frames([frame])[0]

    # ── Data collection ──────────────────────────────────────────────────────
    def collect_step(self, env, obs, env_idx=0):
        """obs is a (vec, frame) tuple, as returned by EnvClient(video=True)."""
        vec, frame = obs
        inp, stores = self._encode_frames([frame])
        return self._collect_one(env, vec, inp, stores[0], env_idx)

    def _collect_one(self, env, vec, inp, store, env_idx):
        self.steps_done += 1
        vec_t = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        state = self.world_model.encode_step(
            self._env_state[env_idx], self._env_prev_act[env_idx], inp, vec_t)
        action, action_onehot = self._act_from_state(state, greedy=False)

        next_obs, reward, done, info = env.step(action)
        self._real_reward_ema += (float(reward) - self._real_reward_ema) / 2000.0
        if self.priv_aux:
            store = dict(store, priv=self._priv_or_zeros(self._env_priv[env_idx]))
            self._env_priv[env_idx] = None if done else info.get("privileged")
        self.buffer.add_obs(store, vec, action, reward, done)

        if done:
            self._env_state[env_idx]    = self._initial_state()
            self._env_prev_act[env_idx] = self._zero_action()
        else:
            self._env_state[env_idx]    = state
            self._env_prev_act[env_idx] = action_onehot
        return next_obs, reward, done, info

    @staticmethod
    def _priv_or_zeros(priv):
        return np.zeros(12, np.float32) if priv is None else np.asarray(priv, np.float32)

    def collect_steps(self, envs, obs_all):
        """Override BaseAgent's default: obs_all is a Python list of (vec, frame)
        tuples (not a stackable ndarray), and each env needs its own env_idx for
        stateful RSSM tracking."""
        n_envs = len(envs)
        next_obs_all = [None] * n_envs
        rewards = np.zeros(n_envs, dtype=np.float32)
        dones   = np.zeros(n_envs, dtype=np.float32)
        infos   = [None] * n_envs
        # One encoder call for every env's frame (the frozen backbone batches well).
        inps, stores = self._encode_frames([o[1] for o in obs_all])
        for i, env in enumerate(envs):
            next_obs, reward, done, info = self._collect_one(
                env, obs_all[i][0], inps[i:i + 1], stores[i], env_idx=i)
            next_obs_all[i] = next_obs
            rewards[i] = reward
            dones[i] = float(done)
            infos[i] = info
        return next_obs_all, rewards, dones, infos

    def buffer_full(self):
        # The replay buffer only samples finished episodes: with 900-step episodes
        # on 3 envs, none may have ended by LEARNING_STARTS, so wait for one.
        return (len(self.buffer) >= self.cfg.LEARNING_STARTS
                and self.buffer.has_samples()
                and self.steps_done % self.cfg.REAL_TRAIN_EVERY == 0)

    def reset_eval_state(self):
        """Call between evaluation episodes when using select_action() directly."""
        self._eval_state = None
        self._eval_prev_act = None

    def select_action(self, obs, greedy=False):
        """obs is a (vec, frame) tuple. Maintains one persistent internal state
        across calls (see class docstring) — call reset_eval_state() at episode
        boundaries."""
        if self._eval_state is None:
            self._eval_state = self._initial_state()
            self._eval_prev_act = self._zero_action()
        vec, frame = obs
        vec_t = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self.world_model.encode_step(
            self._eval_state, self._eval_prev_act, self.model_input(frame), vec_t)
        if getattr(self.cfg, "PLAN_AT_ACT", False):
            action, action_onehot, _ = self._plan_action(state)
        else:
            action, action_onehot = self._act_from_state(state, greedy=greedy)
        self._eval_state = state
        self._eval_prev_act = action_onehot
        return action

    # ── Update: train the RSSM, then train behaviour in imagination ─────────
    def update(self, last_obs=None):
        logs = {}
        posts, mask, actions, n_demo = None, None, None, 0
        for _ in range(self.cfg.WM_GRAD_STEPS):
            wm_logs, posts, mask, actions, n_demo = self._train_world_model()
            logs.update(wm_logs)
        for _ in range(self.cfg.IMAG_UPDATES):
            logs.update(self._imagine_and_train(posts, mask, n_demo))
        if self.demo_buffer is not None:
            logs.update(self._demo_bc_update(posts, mask, actions, n_demo))
        logs["lr"] = self.current_lr
        logs["entropy_coef"] = self.current_entropy_coef
        return logs

    def _sample_batch(self, batch_size, demo_only=False):
        """Real sequences, with DEMO_WM_FRACTION of rows from demos when loaded.
        Demo rows are always LAST. Returns (numpy batch, n_demo_rows)."""
        have_demo = self.demo_buffer is not None and self.demo_buffer.has_samples()
        have_hz = (not demo_only and self.hazard_buffer is not None and self.hazard_buffer.has_samples()
                   and self.buffer.has_samples())
        n_hz = 0
        if have_hz:
            n_hz = max(1, int(round(batch_size * getattr(self.cfg, "HAZARD_WM_FRACTION", 0.1))))
        if not have_demo:
            if n_hz:
                parts = [self.buffer.sample_sequences(batch_size - n_hz),
                         self.hazard_buffer.sample_sequences(n_hz)]
                return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}, 0
            return self.buffer.sample_sequences(batch_size), 0
        if demo_only or not self.buffer.has_samples():
            n_demo = batch_size
        else:
            frac = getattr(self.cfg, "DEMO_WM_FRACTION", 0.1)
            n_demo = min(batch_size - 1 - n_hz, max(1, int(round(batch_size * frac))))
        parts = []
        if batch_size - n_demo - n_hz > 0:
            parts.append(self.buffer.sample_sequences(batch_size - n_demo - n_hz))
        if n_hz:
            parts.append(self.hazard_buffer.sample_sequences(n_hz))
        parts.append(self.demo_buffer.sample_sequences(n_demo))     # demo rows LAST
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}, n_demo

    def _train_world_model(self, demo_only=False):
        batch, n_demo = self._sample_batch(self.cfg.VWM_SEQ_BATCH, demo_only)
        frames  = torch.as_tensor(batch[self.obs_key], device=self.device)
        thumbs  = (torch.as_tensor(batch["thumbs"], device=self.device)
                   if "thumbs" in batch else None)
        priv    = (torch.as_tensor(batch["priv"], device=self.device)
                   if "priv" in batch else None)
        vecs    = torch.as_tensor(batch["vecs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones   = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        mask    = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)

        self.wm_optimizer.zero_grad()
        loss, metrics, posts = self.world_model.loss(
            frames, vecs, actions, rewards, dones, mask, thumbs=thumbs, priv=priv)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.WM_MAX_GRAD_NORM)
        self.wm_optimizer.step()
        return metrics, posts, mask, actions, n_demo

    def pretrain_world_model(self, n_steps, log_every=100):
        """Fit the world model on demos only, before the first live step."""
        if self.demo_buffer is None or not self.demo_buffer.has_samples():
            print("pretrain_world_model: no usable demos loaded (pass --demo-path) — skipping")
            return
        print("Pretraining world model on demos for {0} steps...".format(n_steps))
        for i in range(1, n_steps + 1):
            metrics = self._train_world_model(demo_only=True)[0]
            if i % log_every == 0 or i == n_steps:
                print("  [pretrain {0:>5}/{1}] {2}".format(i, n_steps, " | ".join(
                    "{0}:{1:.4f}".format(k, v) for k, v in metrics.items())))

    def pretrain_actor_bc(self, n_steps, log_every=250):
        """Behaviour-cloning warm start of the actor on the demos, from the world
        model's posterior features of demo clips (the counterpart of
        PPOVision.pretrain_from_demos). Each step also fits the world model on
        the same demo batch, so the features keep improving. Coefficient 1."""
        if self.demo_buffer is None or not self.demo_buffer.has_samples():
            print("pretrain_actor_bc: no usable demos loaded (pass --demo-path) — skipping")
            return
        freeze = bool(getattr(self.cfg, "BC_PRETRAIN_FREEZE_WM", False))
        print("Behaviour-cloning warm start (actor): {0} updates{1}".format(
            n_steps, " (world model frozen)" if freeze else " (co-training the world model)"))
        saved = self._demo_bc_coef
        self._demo_bc_coef = 1.0
        try:
            for i in range(1, n_steps + 1):
                if freeze:
                    posts, mask, actions, n_demo = self._demo_posteriors()
                else:
                    _, posts, mask, actions, n_demo = self._train_world_model(demo_only=True)
                logs = self._demo_bc_update(posts, mask, actions, n_demo)
                if i % log_every == 0 or i == n_steps:
                    print("  [bc {0:>5}/{1}] loss {2:.4f} | match teacher {3:.0%}".format(
                        i, n_steps, logs["bc_loss"], logs["bc_match"]))
        finally:
            self._demo_bc_coef = saved

    @torch.no_grad()
    def _demo_posteriors(self):
        """Posterior states of a demo-only batch with the world model untouched
        (for the frozen-WM behaviour-cloning warm start)."""
        batch, n_demo = self._sample_batch(self.cfg.VWM_SEQ_BATCH, demo_only=True)
        frames  = torch.as_tensor(batch[self.obs_key], device=self.device)
        vecs    = torch.as_tensor(batch["vecs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        mask    = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)
        embed, _ = self.world_model._embed(frames, vecs)
        prev = self.world_model._shift_actions(F.one_hot(actions, self.n_actions).float())
        posts, _ = self.world_model.rssm.observe(embed, prev)
        return posts, mask, actions, n_demo

    def _imagine_and_train(self, posts, mask, n_demo=0):
        """Roll the frozen RSSM prior-only from real posterior start states;
        train the latent actor-critic on Dreamer lambda-returns. Mirrors
        dreamer.py's _imagine_and_train structure (no uncertainty machinery in
        v1 — see docs/world_model/report.md for the vector-side ensemble study
        this intentionally does not replicate yet)."""
        H     = self.cfg.IMAG_HORIZON
        gamma = self.cfg.GAMMA
        lam   = self.cfg.IMAG_LAMBDA
        rssm  = self.world_model.rssm

        flat_mask  = mask.reshape(-1) > 0.5
        valid_idx  = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return {}
        rows, seq = mask.shape
        start_frac = getattr(self.cfg, "DEMO_START_FRACTION", 0.0)
        n_d = 0
        if 0 < n_demo < rows and start_frac > 0:
            # DEMO_START_FRACTION of starts from the demo rows (the last n_demo).
            is_demo = (valid_idx // seq) >= (rows - n_demo)
            demo_idx, real_idx = valid_idx[is_demo], valid_idx[~is_demo]
            n_d = int(round(self.cfg.IMAG_BATCH * start_frac))
            if demo_idx.numel() == 0 or real_idx.numel() == 0:
                n_d = self.cfg.IMAG_BATCH if real_idx.numel() == 0 else 0
            picks = []
            if n_d > 0:
                picks.append(demo_idx[torch.randint(0, demo_idx.numel(), (n_d,), device=self.device)])
            if self.cfg.IMAG_BATCH - n_d > 0:
                picks.append(real_idx[torch.randint(0, real_idx.numel(),
                                                    (self.cfg.IMAG_BATCH - n_d,), device=self.device)])
            sel = torch.cat(picks)
        else:
            n_d = 0
            sel = valid_idx[torch.randint(0, valid_idx.numel(), (self.cfg.IMAG_BATCH,), device=self.device)]
        state = {k: v.reshape(-1, v.shape[-1])[sel].detach() for k, v in posts.items()}
        B = state["stoch"].shape[0]

        pessimism = float(getattr(self.cfg, "IMAG_REWARD_PESSIMISM", 0.0))
        log_probs, entropies, values, rewards, conts, feats, r_stds = [], [], [], [], [], [], []
        for _ in range(H):
            feat = rssm.get_feat(state)
            logits, value = self.model(feat)           # actor-critic forward (grad)
            dist = Categorical(logits=logits)
            a = dist.sample()
            a_onehot = F.one_hot(a, self.n_actions).float()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            values.append(value.squeeze(-1))
            feats.append(feat.detach())
            with torch.no_grad():                       # world model frozen here
                r, cont, r_std, state = self.world_model.step_reward_cont(
                    state, a_onehot, pessimism=pessimism, return_std=True)
            rewards.append(r)
            conts.append(cont)
            r_stds.append(r_std)

        # Slow critic (IMAG_CRITIC_EMA = tau > 0): bootstrap targets come from an
        # exponentially averaged copy of the critic, so the critic no longer
        # chases its own moving estimates (DreamerV2/V3).
        tau = float(getattr(self.cfg, "IMAG_CRITIC_EMA", 0.0))
        target_net = self._slow_critic() if tau > 0 else self.model
        with torch.no_grad():
            _, last_value = target_net(rssm.get_feat(state))
            if tau > 0:
                slow_values = torch.stack([target_net(f)[1].squeeze(-1) for f in feats])
        last_value = last_value.squeeze(-1)

        log_probs = torch.stack(log_probs)              # (H, B)
        entropies = torch.stack(entropies)
        values    = torch.stack(values)                 # (H, B) — requires grad
        rewards   = torch.stack(rewards)                # (H, B) — detached
        conts     = torch.stack(conts)                  # (H, B) — detached

        values_det = slow_values if tau > 0 else values.detach()
        returns = torch.zeros_like(values_det)
        g = last_value
        for t in reversed(range(H)):
            next_v = values_det[t + 1] if t < H - 1 else last_value
            disc = gamma * conts[t]
            g = rewards[t] + disc * ((1.0 - lam) * next_v + lam * g)
            returns[t] = g

        advantage = returns - values_det                # detached -> REINFORCE baseline
        # Return normalisation (IMAG_RETURN_NORM, DreamerV3): divide advantages by
        # an EMA of the 5th-95th percentile range of the returns (at least 1), so
        # the entropy bonus keeps the same weight whether returns are ~1 or ~100.
        if getattr(self.cfg, "IMAG_RETURN_NORM", False):
            flat = returns.detach().flatten()
            rng = (torch.quantile(flat, 0.95) - torch.quantile(flat, 0.05)).item()
            self._ret_scale = rng if getattr(self, "_ret_scale", None) is None else \
                0.99 * self._ret_scale + 0.01 * rng
            advantage = advantage / max(1.0, self._ret_scale)

        actor_loss  = -(log_probs * advantage).mean() - self.current_entropy_coef * entropies.mean()
        critic_loss = 0.5 * ((values - returns) ** 2).mean()
        loss = actor_loss + self.cfg.IMAG_VALUE_COEF * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()
        if tau > 0:
            with torch.no_grad():
                for p_slow, p in zip(self._slow.parameters(), self.model.parameters()):
                    p_slow.mul_(1.0 - tau).add_(p, alpha=tau)

        # Imagined reward split by where the rollout started (demo vs real
        # replay) and the ensemble's reward spread: the diagnostics for "the
        # actor is fooling the world model".
        B = rewards.shape[1]
        r_real = rewards[:, n_d:].mean().item() if n_d < B else 0.0
        r_demo = rewards[:, :n_d].mean().item() if n_d > 0 else 0.0
        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropies.mean().item(),
            "imag_return": returns.mean().item(),
            "imag_reward": rewards.mean().item(),
            # Optimism gap: imagined reward/step (real starts) minus the EMA of the
            # reward the agent actually collects. The one number to watch.
            "real_reward_ema": self._real_reward_ema,
            "optimism_gap": r_real - self._real_reward_ema,
            "imag_reward_real": r_real,
            "imag_reward_demo": r_demo,
            "imag_reward_std":  torch.stack(r_stds).mean().item(),
            "ret_scale":   float(getattr(self, "_ret_scale", None) or 0.0),
        }

    def _slow_critic(self):
        """Lazily created EMA copy of the actor-critic (only its value output is
        used). Rebuilt from the live model after a checkpoint load."""
        if getattr(self, "_slow", None) is None:
            import copy
            self._slow = copy.deepcopy(self.model)
            for p in self._slow.parameters():
                p.requires_grad_(False)
        return self._slow

    def _demo_bc_update(self, posts, mask, actions, n_demo):
        """Behaviour-cloning anchor: cross-entropy of the actor on the demo rows'
        (detached) posterior features vs the demonstrator's actions. Decays with
        training progress (set_progress) so the agent can surpass the teacher.
        Always returns the same keys (the update CSV header is fixed)."""
        if self._demo_bc_coef <= 0.0 or n_demo == 0 or posts is None:
            return {"bc_loss": 0.0, "bc_coef": self._demo_bc_coef, "bc_match": 0.0}
        rows = mask.shape[0]
        demo = slice(rows - n_demo, rows)
        feat = self.world_model.rssm.get_feat({k: v[demo] for k, v in posts.items()})
        keep = mask[demo].reshape(-1) > 0.5
        logits, _ = self.model(feat.reshape(-1, feat.shape[-1]))
        target = actions[demo].reshape(-1)[keep]
        bc_loss = F.cross_entropy(logits[keep], target)
        self.optimizer.zero_grad()
        (self._demo_bc_coef * bc_loss).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()
        match = (logits[keep].argmax(-1) == target).float().mean().item()
        return {"bc_loss": bc_loss.item(), "bc_coef": self._demo_bc_coef, "bc_match": match}

    def _load_demos(self, path, outcomes=None, assign=True):
        """Frames demo directory (utils/demo_io.py) -> self.demo_buffer (assign=True)
        or a returned buffer. `outcomes`: keep only these episode outcomes
        (default: "killed" only when DEMO_ONLY_SUCCESS, else all)."""
        from utils.demo_io import is_frame_demo_dir, load_index, load_episode, episode_features
        if not is_frame_demo_dir(path):
            print("  WARNING: dreamer_video demos must be a frames demo directory "
                  "(utils/demo_io.py); {0} is not — demos disabled".format(path))
            return None
        index = load_index(path)
        total = sum(e["steps"] for e in index["episodes"])
        buf = SequenceReplayBuffer(total + 1, self.cfg.WM_SEQ_LEN, self.cfg.VWM_MIN_SEQ)
        want = (self.cfg.VIDEO_HEIGHT, self.cfg.VIDEO_WIDTH, self.cfg.VIDEO_CHANNELS)
        skipped = 0
        only_success = getattr(self.cfg, "DEMO_ONLY_SUCCESS", False)
        if outcomes is None and only_success:
            outcomes = ("killed",)
        for entry in index["episodes"]:
            if outcomes is not None and entry.get("outcome") not in outcomes:
                continue
            ep = load_episode(path, entry, frames=not self.feature_mode)
            if (ep["vecs"].shape[1] != self.cfg.INPUT_SIZE
                    or int(ep["actions"].max()) >= self.n_actions):
                skipped += 1
                continue
            if self.feature_mode:
                feats, thumbs = episode_features(path, entry, self.frozen_encoder, self.thumb_size)
                obs = {"feats": feats, "thumbs": thumbs}
            else:
                if tuple(ep["frames"].shape[1:]) != want:
                    skipped += 1
                    continue
                obs = {"frames": ep["frames"]}
            if self.priv_aux:
                obs["priv"] = ep.get("privileged", np.zeros((len(ep["actions"]), 12), np.float32))
            buf.add_episode(obs, ep["vecs"], ep["actions"], ep["rewards"], ep["dones"])
        if assign:
            self.demo_buffer = buf
        kills = sum(1 for e in index["episodes"] if e.get("outcome") == "killed")
        print("  Loaded {0} demo episodes ({1} steps, {2} kills in set{3}) from {4}{5}".format(
            len(buf.episodes), len(buf), kills,
            ", outcomes " + "/".join(outcomes) if outcomes else "", path,
            "  [skipped {0}: obs/action/frame shape mismatch]".format(skipped) if skipped else ""))
        return buf

    # ── Optional LR / entropy schedule (train.py calls this if present) ─────
    def set_progress(self, fraction):
        fraction = max(0.0, min(1.0, fraction))
        if getattr(self.cfg, "LR_DECAY", False):
            self.current_lr = self.initial_lr + (self.cfg.LR_END - self.initial_lr) * fraction
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr
        if getattr(self.cfg, "ENTROPY_DECAY", False):
            end = getattr(self.cfg, "ENTROPY_COEF_END", self.initial_entropy_coef)
            self.current_entropy_coef = self.initial_entropy_coef + (end - self.initial_entropy_coef) * fraction
        # Decay the BC anchor so the agent can eventually improve beyond the demonstrator
        # (DEMO_BC_DECAY=False keeps it constant).
        if self.demo_buffer is not None and getattr(self.cfg, "DEMO_BC_DECAY", True):
            self._demo_bc_coef = getattr(self.cfg, "DEMO_BC_COEF", 0.0) * (1.0 - fraction)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def _extra_state(self):
        state = {
            "video_world_model_state": self.world_model.state_dict(),
            "vwm_optimizer_state":     self.wm_optimizer.state_dict(),
            "steps_done": self.steps_done,
            "current_lr": self.current_lr,
            "current_entropy_coef": self.current_entropy_coef,
        }
        if self.feature_mode:
            # The world model is only valid for the encoder it was trained on.
            state["vis_encoder"] = [self.cfg.VIS_ENCODER, self.cfg.VIS_POOL_GRID]
        return state

    def _load_extra_state(self, state):
        if self.feature_mode and "vis_encoder" in state:
            want = [self.cfg.VIS_ENCODER, self.cfg.VIS_POOL_GRID]
            if list(state["vis_encoder"]) != want:
                raise ValueError("Checkpoint was trained on encoder {0}, config uses {1}".format(
                    state["vis_encoder"], want))
        if "video_world_model_state" in state:
            self.world_model.load_state_dict(state["video_world_model_state"])
        if "vwm_optimizer_state" in state:
            self.wm_optimizer.load_state_dict(state["vwm_optimizer_state"])
        self.steps_done = state.get("steps_done", 0)
        self.current_lr = state.get("current_lr", self.initial_lr)
        self.current_entropy_coef = state.get("current_entropy_coef", self.initial_entropy_coef)
