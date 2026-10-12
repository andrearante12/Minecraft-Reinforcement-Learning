"""
visualization/activation_probe.py
---------------------------------
Model-agnostic forward-hook manager for capturing intermediate activations
during a forward pass, plus a helper that wires up the hooks for the
multi-stream ActorCritic network.

The probe is deliberately generic (it imports no model class) so the same
machinery extends to the world-model network later — see
``build_world_model_probe`` notes in the live-viz plan.

Usage:
    from visualization.activation_probe import build_actor_critic_probe

    with build_actor_critic_probe(model) as probe:
        action = agent.select_action(obs, greedy=True)   # forward → hooks fire
        snap   = probe.snapshot()                         # {name: np.ndarray}
    # hooks are removed on exit
"""


class ActivationProbe:
    """Registers forward hooks on a set of named modules and keeps the most
    recent output tensor for each. Capture is O(1) per hook (just a detach);
    the per-step copy to CPU/numpy happens lazily in ``snapshot()``.
    """

    def __init__(self, layers):
        """layers: list of (name, nn.Module). One forward hook per module."""
        self._handles = []
        self.latest = {}  # name -> torch.Tensor (last forward output, detached)
        for name, module in layers:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # Some modules could return tuples; we only ever hook single-tensor
            # outputs here, but guard anyway.
            if isinstance(output, tuple):
                output = output[0]
            self.latest[name] = output.detach()
        return hook

    def snapshot(self, batch_index=0):
        """Return {name: 1-D numpy array} for the given batch row.

        Call once per step, after the forward pass. Tensors are moved to CPU
        and cast to float here (not in the hook) to keep the hook cheap.
        """
        snap = {}
        for name, tensor in self.latest.items():
            t = tensor
            if t.dim() >= 1:
                t = t[batch_index]
            snap[name] = t.reshape(-1).float().cpu().numpy()
        return snap

    def remove(self):
        """Remove all hooks. Safe to call more than once."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always remove hooks, even on error, so a reused model isn't left
        # with leaked hooks.
        self.remove()
        return False


# ── Per-model hook wiring ──────────────────────────────────────────────────
#
# ActorCritic layout (malmo/rl/models/actor_critic.py):
#   *_stream = [Linear, LayerNorm, ReLU, Linear, LayerNorm, ReLU]  -> [-1] = final ReLU
#   *_head   = [Linear, LayerNorm, ReLU, Linear]                   -> [2] = hidden ReLU,
#                                                                    [3] = output Linear

# Layer metadata shared with net_schema so the schema/frame stay in sync with
# the actual hook targets. Each entry: (name, kind), where kind drives how the
# value is normalized/rendered.
ACTOR_CRITIC_LAYERS = [
    ("proprio",       "relu"),    # proprio stream final ReLU
    ("goal",          "relu"),    # goal-delta stream final ReLU
    ("voxel",         "relu"),    # voxel/terrain stream final ReLU
    ("actor_hidden",  "relu"),    # actor head hidden ReLU
    ("actor_logits",  "logits"),  # raw action logits (output neurons)
    ("critic_hidden", "relu"),    # critic head hidden ReLU
    ("value",         "scalar"),  # scalar state value
]

SB3_MLP_LAYERS = [
    ("actor_hidden",  "activation"),
    ("actor_logits",  "logits"),
    ("critic_hidden", "activation"),
    ("value",         "scalar"),
]


def build_actor_critic_probe(model):
    """Return an ActivationProbe hooked into the 7 informative points of an
    ActorCritic instance. Returns the probe (use it as a context manager).
    """
    layers = [
        ("proprio",       model.proprio_stream[-1]),
        ("goal",          model.goal_stream[-1]),
        ("voxel",         model.voxel_stream[-1]),
        ("actor_hidden",  model.actor[2]),
        ("actor_logits",  model.actor[3]),
        ("critic_hidden", model.critic[2]),
        ("value",         model.critic[3]),
    ]
    return ActivationProbe(layers)


def build_sb3_mlp_probe(policy):
    """Capture the real actor/critic outputs of an SB3 ``MlpPolicy``.

    Hooking the complete policy/value ``Sequential`` modules avoids assuming a
    particular hidden-layer count or activation function. The action and value
    heads then expose the exact logits and scalar used for the decision.
    """
    extractor = policy.mlp_extractor
    layers = [
        ("actor_hidden",  extractor.policy_net),
        ("actor_logits",  policy.action_net),
        ("critic_hidden", extractor.value_net),
        ("value",         policy.value_net),
    ]
    return ActivationProbe(layers)
