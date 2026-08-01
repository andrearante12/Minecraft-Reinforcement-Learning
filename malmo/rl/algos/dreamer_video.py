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

        # Per-env RSSM posterior state + previous action (for stateful acting).
        self._env_state    = [self._initial_state() for _ in range(n_envs)]
        self._env_prev_act = [self._zero_action() for _ in range(n_envs)]

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

    # ── Data collection ──────────────────────────────────────────────────────
    def collect_step(self, env, obs, env_idx=0):
        """obs is a (vec, frame) tuple, as returned by EnvClient(video=True)."""
        self.steps_done += 1
        vec, frame = obs
        vec_t   = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        frame_t = torch.as_tensor(frame, device=self.device).unsqueeze(0)

        state = self.world_model.encode_step(
            self._env_state[env_idx], self._env_prev_act[env_idx], frame_t, vec_t)
        action, action_onehot = self._act_from_state(state, greedy=False)

        next_obs, reward, done, info = env.step(action)
        self.buffer.add(frame, vec, action, reward, done)

        if done:
            self._env_state[env_idx]    = self._initial_state()
            self._env_prev_act[env_idx] = self._zero_action()
        else:
            self._env_state[env_idx]    = state
            self._env_prev_act[env_idx] = action_onehot
        return next_obs, reward, done, info

    def collect_steps(self, envs, obs_all):
        """Override BaseAgent's default: obs_all is a Python list of (vec, frame)
        tuples (not a stackable ndarray), and each env needs its own env_idx for
        stateful RSSM tracking."""
        n_envs = len(envs)
        next_obs_all = [None] * n_envs
        rewards = np.zeros(n_envs, dtype=np.float32)
        dones   = np.zeros(n_envs, dtype=np.float32)
        infos   = [None] * n_envs
        for i, env in enumerate(envs):
            next_obs, reward, done, info = self.collect_step(env, obs_all[i], env_idx=i)
            next_obs_all[i] = next_obs
            rewards[i] = reward
            dones[i] = float(done)
            infos[i] = info
        return next_obs_all, rewards, dones, infos

    def buffer_full(self):
        return (len(self.buffer) >= self.cfg.LEARNING_STARTS
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
        vec_t   = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        frame_t = torch.as_tensor(frame, device=self.device).unsqueeze(0)
        state = self.world_model.encode_step(self._eval_state, self._eval_prev_act, frame_t, vec_t)
        action, action_onehot = self._act_from_state(state, greedy=greedy)
        self._eval_state = state
        self._eval_prev_act = action_onehot
        return action

    # ── Update: train the RSSM, then train behaviour in imagination ─────────
    def update(self, last_obs=None):
        logs = {}
        posts, mask = None, None
        for _ in range(self.cfg.WM_GRAD_STEPS):
            wm_logs, posts, mask = self._train_world_model()
            logs.update(wm_logs)
        for _ in range(self.cfg.IMAG_UPDATES):
            logs.update(self._imagine_and_train(posts, mask))
        logs["lr"] = self.current_lr
        logs["entropy_coef"] = self.current_entropy_coef
        return logs

    def _train_world_model(self):
        batch = self.buffer.sample_sequences(self.cfg.VWM_SEQ_BATCH)
        frames  = torch.as_tensor(batch["frames"], device=self.device)
        vecs    = torch.as_tensor(batch["vecs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones   = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        mask    = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)

        self.wm_optimizer.zero_grad()
        loss, metrics, posts = self.world_model.loss(frames, vecs, actions, rewards, dones, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.WM_MAX_GRAD_NORM)
        self.wm_optimizer.step()
        return metrics, posts, mask

    def _imagine_and_train(self, posts, mask):
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
        sel = valid_idx[torch.randint(0, valid_idx.numel(), (self.cfg.IMAG_BATCH,), device=self.device)]
        state = {k: v.reshape(-1, v.shape[-1])[sel].detach() for k, v in posts.items()}
        B = state["stoch"].shape[0]

        log_probs, entropies, values, rewards, conts = [], [], [], [], []
        for _ in range(H):
            feat = rssm.get_feat(state)
            logits, value = self.model(feat)           # actor-critic forward (grad)
            dist = Categorical(logits=logits)
            a = dist.sample()
            a_onehot = F.one_hot(a, self.n_actions).float()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            values.append(value.squeeze(-1))
            with torch.no_grad():                       # world model frozen here
                r, cont = self.world_model.predict_reward_cont(feat, a_onehot)
                state = rssm.img_step(state, a_onehot)
            rewards.append(r)
            conts.append(cont)

        with torch.no_grad():
            _, last_value = self.model(rssm.get_feat(state))
        last_value = last_value.squeeze(-1)

        log_probs = torch.stack(log_probs)              # (H, B)
        entropies = torch.stack(entropies)
        values    = torch.stack(values)                 # (H, B) — requires grad
        rewards   = torch.stack(rewards)                # (H, B) — detached
        conts     = torch.stack(conts)                  # (H, B) — detached

        values_det = values.detach()
        returns = torch.zeros_like(values_det)
        g = last_value
        for t in reversed(range(H)):
            next_v = values_det[t + 1] if t < H - 1 else last_value
            disc = gamma * conts[t]
            g = rewards[t] + disc * ((1.0 - lam) * next_v + lam * g)
            returns[t] = g

        advantage = returns - values_det                # detached -> REINFORCE baseline

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

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def _extra_state(self):
        return {
            "video_world_model_state": self.world_model.state_dict(),
            "vwm_optimizer_state":     self.wm_optimizer.state_dict(),
            "steps_done": self.steps_done,
            "current_lr": self.current_lr,
            "current_entropy_coef": self.current_entropy_coef,
        }

    def _load_extra_state(self, state):
        if "video_world_model_state" in state:
            self.world_model.load_state_dict(state["video_world_model_state"])
        if "vwm_optimizer_state" in state:
            self.wm_optimizer.load_state_dict(state["vwm_optimizer_state"])
        self.steps_done = state.get("steps_done", 0)
        self.current_lr = state.get("current_lr", self.initial_lr)
        self.current_entropy_coef = state.get("current_entropy_coef", self.initial_entropy_coef)
