"""
visualization/net_schema.py
---------------------------
Turns raw activation snapshots (from ActivationProbe) into the two JSON
payloads the browser frontend consumes:

  * ``build_schema(cfg, model)`` — the *static* topology, sent once on connect:
    layer ids/labels/sizes, how many representative cells to draw per layer,
    the action labels, and the input-readout fields.

  * ``FrameBuilder.build(snapshot, action, norm_input)`` — the *per-step* frame:
    activations down-sampled to representative cells and normalized to [0, 1]
    (stable across steps via a per-layer EMA scale), action probabilities
    (softmax of logits), the scalar value mapped against a running range, and
    a small human-readable input readout.

Keeping all numeric work here means the frontend only ever renders values that
are already in [0, 1] (plus a couple of raw numbers shown as text).
"""

import numpy as np

from visualization.activation_probe import ACTOR_CRITIC_LAYERS

# Which captured tensors are rendered as banks of neuron cells (vs. handled
# specially: actor_logits -> probability bars, value -> gauge).
_CELL_LAYERS = [name for name, kind in ACTOR_CRITIC_LAYERS if kind == "relu"]

# Display group + human label for each cell layer.
_LAYER_META = {
    "proprio":       ("stream", "Proprio"),
    "goal":          ("stream", "Goal Δ"),
    "voxel":         ("stream", "Voxel"),
    "actor_hidden":  ("head",   "Actor hidden"),
    "critic_hidden": ("head",   "Critic hidden"),
}

# Max representative cells drawn per layer. Layers smaller than this show all
# their neurons; larger layers are evenly sub-sampled (fixed indices, so each
# cell stays tied to one neuron and can tween smoothly).
_DISPLAY_CELLS = 16


def _display_indices(size, n=_DISPLAY_CELLS):
    """Fixed, evenly-spaced neuron indices for a layer (stable across steps)."""
    if size <= n:
        return list(range(size))
    return [int(round(i)) for i in np.linspace(0, size - 1, n)]


def _layer_sizes(model):
    """Neuron count per cell-layer, read from the model's first Linear in each
    block (out_features). Avoids hard-coding 64/128/256.
        stream = [Linear, LayerNorm, ReLU, Linear, LayerNorm, ReLU]
        head   = [Linear, LayerNorm, ReLU, Linear]
    """
    return {
        "proprio":       model.proprio_stream[0].out_features,
        "goal":          model.goal_stream[0].out_features,
        "voxel":         model.voxel_stream[0].out_features,
        "actor_hidden":  model.actor[0].out_features,
        "critic_hidden": model.critic[0].out_features,
    }


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float64)


def build_schema(cfg, model):
    """Static topology for the frontend. Sizes are read from the live model so
    nothing is hard-coded (works for parkour, bridging, etc.).
    """
    sizes = _layer_sizes(model)
    layers = []
    for name in _CELL_LAYERS:
        group, label = _LAYER_META[name]
        size = int(sizes[name])
        layers.append({
            "id": name,
            "label": label,
            "group": group,
            "size": size,
            "display": len(_display_indices(size)),
        })

    action_names = [a[0] for a in cfg.ACTIONS]

    # Input readout: the first 6 proprio dims are shared across tasks
    # (onGround, yaw, pitch, delta_y, delta_x, delta_z), then the goal delta.
    input_fields = []
    if model._proprio_size >= 6:
        input_fields = [
            {"key": "on_ground", "label": "On ground"},
            {"key": "vel",       "label": "Velocity (Δy,Δx,Δz)", "dims": 3},
            {"key": "goal",      "label": "Goal Δ (dx,dy,dz)",   "dims": 3},
        ]

    return {
        "type": "schema",
        "env": getattr(cfg, "ENV_NAME", type(cfg).__name__),
        "actions": action_names,
        "layers": layers,
        "input_fields": input_fields,
    }


def build_sb3_schema(cfg, policy):
    """Describe the flat actor/critic MLP used by SB3 bridging checkpoints."""
    actor_size = int(policy.mlp_extractor.latent_dim_pi)
    critic_size = int(policy.mlp_extractor.latent_dim_vf)
    return {
        "type": "schema",
        "env": getattr(cfg, "ENV_NAME", "bridging"),
        "backend": "sb3_mlp",
        "actions": [a[0] for a in cfg.ACTIONS],
        "layers": [
            {
                "id": "actor_hidden", "label": "Actor MLP", "group": "stream",
                "size": actor_size, "display": len(_display_indices(actor_size)),
            },
            {
                "id": "critic_hidden", "label": "Critic MLP", "group": "stream",
                "size": critic_size, "display": len(_display_indices(critic_size)),
            },
        ],
        "vox": {"cols": int(cfg.GRID_X), "rows": int(cfg.GRID_Z)},
        "input_fields": [
            {"key": "on_ground", "label": "On ground", "kind": "bool"},
            {"key": "inv", "label": "Inventory"},
            {"key": "ray", "label": "Ray hit", "kind": "bool"},
            {"key": "goal", "label": "Goal Δ (dx,dy,dz)", "dims": 3},
            {"key": "vox", "label": "Bridge voxels (top-down)"},
        ],
    }


class FrameBuilder:
    """Stateful: keeps a per-layer EMA scale so neuron brightness is comparable
    across steps, and a running min/max for the value gauge.
    """

    def __init__(self, cfg, model, ema_decay=0.98, eps=1e-3):
        self.cfg = cfg
        self.ema_decay = ema_decay
        self.eps = eps
        self._scale = {}                       # layer -> EMA of p99(|act|)
        self._proprio_size = model._proprio_size
        self._goal_end = model._goal_end
        self._action_names = [a[0] for a in cfg.ACTIONS]
        sizes = _layer_sizes(model)
        self._disp = {name: _display_indices(sizes[name]) for name in _CELL_LAYERS}
        self._vmin = None
        self._vmax = None
        self._step = 0

    def _normalize_layer(self, name, act):
        """Scale a layer's activations to [0, 1] using an EMA of its p99(|act|)."""
        p99 = float(np.percentile(np.abs(act), 99)) if act.size else 0.0
        prev = self._scale.get(name, p99)
        scale = max(self.ema_decay * prev, p99)
        scale = max(scale, self.eps)
        self._scale[name] = scale
        return np.clip(act / scale, 0.0, 1.0)

    def build(self, snapshot, action, norm_input):
        self._step += 1

        activations = {}
        stream_energy = {}
        for name in _CELL_LAYERS:
            act = snapshot[name]
            norm = self._normalize_layer(name, act)
            # representative (display) cells
            idx = self._disp[name]
            activations[name] = [round(float(norm[i]), 4) for i in idx]
            # energy over ALL neurons (drives the stream-node glow)
            stream_energy[name] = round(float(norm.mean()), 4)

        logits = snapshot["actor_logits"]
        probs = _softmax(logits)
        value = float(snapshot["value"][0])

        # running value range (slightly padded so the gauge isn't pinned)
        self._vmin = value if self._vmin is None else min(self._vmin, value)
        self._vmax = value if self._vmax is None else max(self._vmax, value)
        span = max(self._vmax - self._vmin, 1e-6)
        value_norm = (value - self._vmin) / span

        frame = {
            "type": "frame",
            "step": self._step,
            "action": int(action),
            "action_name": self._action_names[int(action)],
            "probs": [round(float(p), 4) for p in probs],
            "logits": [round(float(x), 3) for x in logits],
            "value": round(value, 3),
            "value_norm": round(float(value_norm), 4),
            "value_min": round(float(self._vmin), 3),
            "value_max": round(float(self._vmax), 3),
            "activations": activations,
            "stream_energy": stream_energy,
            "inputs": self._input_readout(norm_input),
        }
        return frame

    def _input_readout(self, x):
        """Pull a few human-meaningful values out of the (normalized) obs."""
        if x is None or self._proprio_size < 6:
            return {}
        x = np.asarray(x, dtype=np.float64)
        return {
            "on_ground": round(float(x[0]), 3),
            # proprio dims 3:6 = delta_y, delta_x, delta_z
            "vel":  [round(float(v), 3) for v in x[3:6]],
            "goal": [round(float(v), 3) for v in x[self._proprio_size:self._goal_end]],
        }


class Sb3BridgingFrameBuilder:
    """Convert real SB3 MLP activations and bridging observations for the site."""

    _CELL_LAYERS = ("actor_hidden", "critic_hidden")

    def __init__(self, cfg, policy, ema_decay=0.98, eps=1e-3):
        self.cfg = cfg
        self.ema_decay = ema_decay
        self.eps = eps
        self._scale = {}
        self._action_names = [a[0] for a in cfg.ACTIONS]
        sizes = {
            "actor_hidden": int(policy.mlp_extractor.latent_dim_pi),
            "critic_hidden": int(policy.mlp_extractor.latent_dim_vf),
        }
        self._disp = {
            name: _display_indices(size) for name, size in sizes.items()
        }
        self._vmin = None
        self._vmax = None
        self._step = 0

    def _normalize_layer(self, name, act):
        """Map signed Tanh/ReLU activations to magnitude in [0, 1]."""
        magnitude = np.abs(np.asarray(act, dtype=np.float64))
        p99 = float(np.percentile(magnitude, 99)) if magnitude.size else 0.0
        previous = self._scale.get(name, p99)
        scale = max(self.ema_decay * previous, p99, self.eps)
        self._scale[name] = scale
        return np.clip(magnitude / scale, 0.0, 1.0)

    def build(self, snapshot, action, obs, reward, outcome, frame_index, step_dt):
        self._step += 1
        activations = {}
        stream_energy = {}
        for name in self._CELL_LAYERS:
            norm = self._normalize_layer(name, snapshot[name])
            activations[name] = [
                round(float(norm[i]), 4) for i in self._disp[name]
            ]
            stream_energy[name] = round(float(norm.mean()), 4)

        logits = np.asarray(snapshot["actor_logits"], dtype=np.float64)
        probs = _softmax(logits)
        value = float(snapshot["value"][0])
        self._vmin = value if self._vmin is None else min(self._vmin, value)
        self._vmax = value if self._vmax is None else max(self._vmax, value)
        span = max(self._vmax - self._vmin, 1e-6)

        return {
            "type": "frame",
            "step": self._step,
            "frame_index": int(frame_index),
            "t_sec": round(float(frame_index) * float(step_dt), 6),
            "action": int(action),
            "action_name": self._action_names[int(action)],
            "probs": [round(float(p), 4) for p in probs],
            "logits": [round(float(x), 4) for x in logits],
            "value": round(value, 4),
            "value_norm": round(float((value - self._vmin) / span), 4),
            "value_min": round(float(self._vmin), 4),
            "value_max": round(float(self._vmax), 4),
            "reward": round(float(reward), 4),
            "outcome": str(outcome),
            "activations": activations,
            "stream_energy": stream_energy,
            "inputs": self._input_readout(obs),
        }

    def _input_readout(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        proprio_end = int(self.cfg.PROPRIOCEPTION_SIZE)
        goal_end = proprio_end + int(self.cfg.GOAL_DELTA_SIZE)
        voxels = x[goal_end:goal_end + int(self.cfg.GRID_SIZE)]
        try:
            top_down = voxels.reshape(
                int(self.cfg.GRID_Y), int(self.cfg.GRID_Z), int(self.cfg.GRID_X)
            ).max(axis=0).reshape(-1)
        except ValueError:
            top_down = np.zeros(int(self.cfg.GRID_X) * int(self.cfg.GRID_Z))
        return {
            "on_ground": round(float(x[0]), 3),
            "inv": round(float(x[9]), 3),
            "ray": round(float(x[10]), 3),
            "goal": [round(float(v), 3) for v in x[proprio_end:goal_end]],
            "vox": [int(v > 0.5) for v in top_down],
        }
