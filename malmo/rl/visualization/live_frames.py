"""
visualization/live_frames.py
----------------------------
Turn one real step (obs + Minecraft camera frame) into the JSON frame the live
split-screen page renders next to the camera. Auto-detected by agent type:

  VectorDreamerFrames  (algos/dreamer.py, or any ActorCritic agent)
      policy π(a|s), V(s), net activations, and — with a world-model ensemble —
      the K members' imagined |pig Δ| over the horizon ("fan") plus the
      aleatoric / epistemic split at the current state.

  VideoDreamerFrames   (algos/dreamer_video.py)
      π / V on the RSSM feature, the posterior decode ("what the model sees"),
      and a prior-only imagined frame strip with predicted reward / continuation.

Everything runs under torch.no_grad(); callers only build frames while a
browser is connected (LiveServer.has_clients), so training pays nothing otherwise.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from visualization.live_server import encode_jpeg
from visualization.net_schema import build_schema, FrameBuilder, _softmax
from visualization.activation_probe import build_actor_critic_probe

FAN_HORIZON_CAP = 15
DREAM_HORIZON = 8
CAM_MAX_WIDTH = 960


class _RunningMax:
    """Slowly-decaying max so gauges use the full [0,1] range but still adapt."""

    def __init__(self, decay=0.995, floor=1e-6):
        self.decay, self.floor, self.value = decay, floor, None

    def __call__(self, x):
        x = float(x)
        self.value = x if self.value is None else max(self.value * self.decay, x)
        return min(1.0, x / max(self.value, self.floor))


class _RelGauge:
    """x / (x + EMA(x)): 0.5 at the recent norm, →1 on spikes, →0 when it drops."""

    def __init__(self, decay=0.98, floor=1e-8):
        self.decay, self.floor, self.ema = decay, floor, None

    def __call__(self, x):
        x = max(float(x), 0.0)
        self.ema = x if self.ema is None else self.decay * self.ema + (1 - self.decay) * x
        return x / max(x + self.ema, self.floor)


def _action_names(cfg):
    return [a[0] for a in cfg.ACTIONS]


def _cam_jpeg(cam):
    return encode_jpeg(cam, quality=78, max_width=CAM_MAX_WIDTH) if cam is not None else None


def make_frame_builder(agent, cfg, env_name=None):
    from algos.dreamer_video import DreamerVideo
    cls = VideoDreamerFrames if isinstance(agent, DreamerVideo) else VectorDreamerFrames
    builder = cls(agent, cfg)
    builder.env_name = env_name or getattr(cfg, "__name__", type(cfg).__name__)
    return builder


class VectorDreamerFrames:
    kind = "vector"

    def __init__(self, agent, cfg):
        self.agent, self.cfg = agent, cfg
        self.model = agent.model
        self.fb = FrameBuilder(cfg, self.model)
        wm = getattr(agent, "world_model", None)
        self.ensemble = wm if hasattr(wm, "members") else None
        self.single_wm = wm if (wm is not None and self.ensemble is None) else None
        self.horizon = min(int(getattr(cfg, "IMAG_HORIZON", 10)), FAN_HORIZON_CAP)
        self.unc_dims = list(getattr(cfg, "WM_UNCERTAINTY_DIMS", (0, 1, 2)))
        self._goal_lo = int(cfg.PROPRIOCEPTION_SIZE)
        self._fan_max = _RunningMax(decay=0.999)
        self._ale_gauge = _RelGauge()
        self._epi_gauge = _RelGauge()

    def schema(self):
        s = build_schema(self.cfg, self.model)
        s["kind"] = self.kind
        s["env"] = self.env_name
        s["has_world_model"] = self.ensemble is not None or self.single_wm is not None
        s["ensemble_k"] = len(self.ensemble.members) if self.ensemble is not None else 1
        return s

    def _device(self):
        return next(self.model.parameters()).device

    def _target_dist(self, obs_t):
        idx = [self._goal_lo + d for d in self.unc_dims]
        return obs_t[:, idx].norm(dim=-1)

    @torch.no_grad()
    def build(self, obs, cam=None, env_idx=0, mode="train"):
        dev = self._device()
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=dev).unsqueeze(0)

        # Forward with hooks attached only for this call — no cost on training forwards.
        with build_actor_critic_probe(self.model) as probe:
            logits, _ = self.model(obs_t)
            snap = probe.snapshot()
        greedy = int(logits.argmax(dim=-1).item())
        frame = self.fb.build(snap, greedy, obs)
        frame["kind"] = self.kind
        frame["cam"] = _cam_jpeg(cam)

        wm = self.ensemble if self.ensemble is not None else self.single_wm
        if wm is None:
            return frame

        if self.ensemble is not None:
            a0 = logits.argmax(dim=-1)
            _, _, _, ale, epi = self.ensemble.predict_with_uncertainty(obs_t, a0)
            ale, epi = float(ale.item()), float(epi.item())
            frame["aleatoric_raw"] = round(ale, 5)
            frame["epistemic_raw"] = round(epi, 5)
            frame["aleatoric"] = round(self._ale_gauge(ale), 4)
            frame["epistemic"] = round(self._epi_gauge(epi), 4)
            members = list(self.ensemble.members)
        else:
            members = [self.single_wm]

        if getattr(self.cfg, "TARGET_MOB", None) is not None:
            frame["topdown"] = self._topdown(obs_t, wm, members)
        else:
            frame["fan"] = self._fan(obs_t, members)
        return frame

    # ── top-down imagination (hunting: absolute positions are in the obs) ────
    def _xz(self, s):
        """(agent_xz, pig_xz or None, yaw_deg) from one obs row (world coords rel. spawn).
        proprio: [1]=yaw/180, [6]=x-spawn_x, [8]=z-spawn_z; target: [0..2]=pig-agent Δ, [11]=visible."""
        v = s[0].cpu().numpy()
        ax, az = float(v[6]), float(v[8])
        g = self._goal_lo
        visible = v[g + 11] > 0.5
        pig = [ax + float(v[g]), az + float(v[g + 2])] if visible else None
        return [ax, az], pig, float(v[1]) * 180.0

    def _topdown(self, obs_t, wm, members):
        """Every member rolls H steps under ONE shared action sequence (picked by the
        actor on the ensemble-mean rollout), so spread between members is purely
        epistemic. Each member's predicted pig-Δ variance is accumulated along the
        path → a 1σ radius per step (aleatoric)."""
        H = self.horizon
        actions, s = [], obs_t
        for _ in range(H):
            lg, _ = self.model(s)
            a = lg.argmax(dim=-1)
            actions.append(a)
            s, _, _ = wm.predict(s, a)

        agent0, pig0, yaw0 = self._xz(obs_t)
        out = []
        g = self._goal_lo
        for m in members:
            s, var = obs_t, 0.0
            a_path, p_path, sig = [agent0], [pig0], [0.0]
            for a in actions:
                hd = m._heads(s, a)
                if hd["d_goal_logvar"] is not None:
                    lv = hd["d_goal_logvar"][0]
                    var += float(torch.exp(lv[0]).item() + torch.exp(lv[2]).item()) / 2.0
                s, _, _ = m.predict(s, a)
                ag = [float(s[0, 6].item()), float(s[0, 8].item())]
                a_path.append(ag)
                p_path.append([ag[0] + float(s[0, g].item()), ag[1] + float(s[0, g + 2].item())])
                sig.append(var ** 0.5)
            r = lambda pts: [[round(p[0], 2), round(p[1], 2)] for p in pts]
            out.append({
                "agent": r(a_path),
                # pig Δ is zeroed while the pig is out of sight → nothing to imagine
                "pig": r(p_path[1:]) if pig0 else None,
                "sigma": [round(v, 3) for v in sig[1:]],
            })

        names = _action_names(self.cfg)
        return {
            "agent": [round(agent0[0], 2), round(agent0[1], 2)],
            "pig": [round(pig0[0], 2), round(pig0[1], 2)] if pig0 else None,
            "yaw": round(yaw0, 1),
            "members": out,
            "plan": [names[int(a.item())] for a in actions],
        }

    def _fan(self, obs_t, members):
        """Non-target envs: each member's imagined |goal Δ| over the horizon, in [0,1]."""
        raw_paths = []
        for m in members:
            s = obs_t
            path = [float(self._target_dist(s).item())]
            for _ in range(self.horizon):
                lg, _ = self.model(s)
                s, _, _ = m.predict(s, lg.argmax(dim=-1))
                path.append(float(self._target_dist(s).item()))
            raw_paths.append(path)
        peak = max(max(p) for p in raw_paths)
        self._fan_max(peak)
        scale = max(self._fan_max.value or peak, 1e-6)
        return [[round(min(1.0, v / scale), 4) for v in p] for p in raw_paths]


class VideoDreamerFrames:
    kind = "video"

    def __init__(self, agent, cfg):
        self.agent, self.cfg = agent, cfg
        self.wm = agent.world_model
        self.model = agent.model
        self.n_actions = cfg.N_ACTIONS
        self.horizon = DREAM_HORIZON
        self._vmin = None
        self._vmax = None

    def schema(self):
        return {
            "type": "schema",
            "kind": self.kind,
            "env": self.env_name,
            "actions": _action_names(self.cfg),
            "layers": [],
            "input_fields": [],
            "dream_horizon": self.horizon,
            "frame_shape": [self.cfg.VIDEO_HEIGHT, self.cfg.VIDEO_WIDTH, self.cfg.VIDEO_CHANNELS],
        }

    def _posterior(self, obs, env_idx, mode):
        """RSSM state that has seen the current frame.

        eval: select_action() already encoded this frame into _eval_state — reuse it.
        train: collect_step() hasn't run yet; encode on a copy of the env's state.
        """
        a = self.agent
        if mode == "eval" and a._eval_state is not None:
            return a._eval_state
        vec, frame = obs
        vec_t = torch.as_tensor(vec, dtype=torch.float32, device=a.device).unsqueeze(0)
        # model_input: pixels as-is, or frozen-encoder tokens (VWM_INPUT="features")
        return self.wm.encode_step(a._env_state[env_idx], a._env_prev_act[env_idx],
                                   a.model_input(frame), vec_t)

    @staticmethod
    def _to_img(decoded):
        return (decoded.squeeze(0) + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def build(self, obs, cam=None, env_idx=0, mode="train"):
        rssm = self.wm.rssm
        state = self._posterior(obs, env_idx, mode)
        feat = rssm.get_feat(state)
        logits, value = self.model(feat)
        probs = _softmax(logits.squeeze(0).cpu().numpy())
        value = float(value.reshape(-1)[0].item())
        self._vmin = value if self._vmin is None else min(self._vmin, value)
        self._vmax = value if self._vmax is None else max(self._vmax, value)
        span = max(self._vmax - self._vmin, 1e-6)
        greedy = int(np.argmax(probs))

        recon, _ = self.wm.decode(feat)

        # Prior-only dream from the current posterior (actor samples its own actions).
        dream, s = [], {k: v.clone() for k, v in state.items()}
        for _ in range(self.horizon):
            f = rssm.get_feat(s)
            lg, _ = self.model(f)
            act = Categorical(logits=lg).sample()
            onehot = F.one_hot(act, self.n_actions).float()
            r, c = self.wm.predict_reward_cont(f, onehot)
            s = rssm.img_step(s, onehot)
            img, _ = self.wm.decode(rssm.get_feat(s))
            dream.append({
                "img": encode_jpeg(self._to_img(img), quality=85),
                "action": _action_names(self.cfg)[int(act.item())],
                "reward": round(float(r.item()), 3),
                "cont": round(float(c.item()), 3),
            })

        real_small = obs[1] if isinstance(obs, (tuple, list)) else None
        return {
            "type": "frame",
            "kind": self.kind,
            "action": greedy,
            "action_name": _action_names(self.cfg)[greedy],
            "probs": [round(float(p), 4) for p in probs],
            "value": round(value, 3),
            "value_norm": round((value - self._vmin) / span, 4),
            "value_min": round(self._vmin, 3),
            "value_max": round(self._vmax, 3),
            "cam": _cam_jpeg(cam if cam is not None else real_small),
            "real_small": encode_jpeg(real_small, quality=90) if real_small is not None else None,
            "recon": encode_jpeg(self._to_img(recon), quality=90),
            "dream": dream,
        }


def annotate(frame, cfg, action, episode, step, ep_reward, mode, last_reward=None, outcome=None):
    """Stamp what actually happened this step onto a built frame (in place)."""
    frame["action"] = int(action)
    frame["action_name"] = _action_names(cfg)[int(action)]
    frame["episode"] = int(episode)
    frame["step"] = int(step)
    frame["ep_reward"] = round(float(ep_reward), 3)
    frame["mode"] = mode
    if last_reward is not None:
        frame["reward"] = round(float(last_reward), 3)
    if outcome is not None:
        frame["outcome"] = outcome
    return frame
