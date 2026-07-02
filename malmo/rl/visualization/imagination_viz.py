"""
visualization/imagination_viz.py
--------------------------------
Visualize what the world model is "imagining".

Because the world model predicts in OBSERVATION SPACE, an imagined rollout is a
sequence of predicted observations that decode to the same {x,y,z,...} step
dicts the existing trajectory renderers consume. This module turns a trained
Dreamer checkpoint's dreams into figures:

  (a) Open-loop overlay  — roll the world model under a real action sequence
      from a real start state; overlay the imagined agent (and target) path
      (dashed) on the real path (solid). Imagination visibly diverges from
      reality as the horizon grows — this is also the prediction-error figure.
  (b) Dream rollout      — let the actor act *inside* the world model; render
      the imagined trajectory (no Malmo / no real data needed).
  (c) Metric traces      — predicted reward / value / continuation over horizon.
  (d) Voxel montage      — the predicted occupancy grid across the horizon.

Decoding is env-agnostic: the agent path is reconstructed by integrating the
predicted per-step velocity (proprio indices [4,3,5] = dx,dy,dz), which every
MalmoRL observation carries. For envs whose target/goal stream holds a relative
position (e.g. hunting: indices [P..P+2] = target-agent delta), a second path is
decoded as agent_pos + delta.

CLI (run in train_env):
    python visualization/imagination_viz.py --env hunting \
        --checkpoint checkpoints/dreamer_hunting_ep200.pt --out /tmp/dream
"""

import os
import sys
import argparse
import numpy as np

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

import torch
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from visualization.traj_renderer import render_trajectory, render_imagined_trajectory


# ── Rollout ──────────────────────────────────────────────────────────────────
def imagine(world_model, model, start_obs, horizon, actions=None, device=None):
    """Roll the world model `horizon` steps from start_obs.

    If `actions` (length-horizon int sequence) is given, those actions are
    replayed open-loop; otherwise the actor samples actions ("dream").

    Returns a dict of numpy arrays:
        obs_seq    (H+1, INPUT_SIZE)   imagined observations (incl. start)
        actions    (H,)               actions taken
        rewards    (H,)               predicted rewards
        values     (H+1,)             critic values along the rollout
        conts      (H,)               predicted continuation probabilities
    """
    device = device or next(model.parameters()).device
    s = torch.as_tensor(np.asarray(start_obs, dtype=np.float32), device=device).unsqueeze(0)

    obs_seq, acts, rewards, values, conts = [s.squeeze(0).cpu().numpy()], [], [], [], []
    with torch.no_grad():
        for t in range(horizon):
            logits, value = model(s)
            values.append(float(value.squeeze(-1).item()))
            if actions is not None:
                a = torch.tensor([int(actions[t])], device=device)
            else:
                a = Categorical(logits=logits).sample()
            s, r, cont = world_model.predict(s, a)
            acts.append(int(a.item()))
            rewards.append(float(r.item()))
            conts.append(float(cont.item()))
            obs_seq.append(s.squeeze(0).cpu().numpy())
        _, last_value = model(s)
        values.append(float(last_value.squeeze(-1).item()))

    return {
        "obs_seq": np.array(obs_seq, dtype=np.float32),
        "actions": np.array(acts, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "values":  np.array(values, dtype=np.float32),
        "conts":   np.array(conts, dtype=np.float32),
    }


# ── Decoding obs → world-space paths ─────────────────────────────────────────
def decode_agent_path(obs_seq, cfg):
    """Reconstruct the agent's world path by integrating predicted velocity.

    Observation velocity lives at proprio indices [3,4,5] = (dy, dx, dz).
    """
    spawn = np.array(cfg.SPAWN, dtype=np.float32)
    pos = spawn.copy()
    steps = [{"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}]
    for o in obs_seq[1:]:
        dy, dx, dz = float(o[3]), float(o[4]), float(o[5])
        pos = pos + np.array([dx, dy, dz], dtype=np.float32)
        steps.append({"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])})
    return steps


def decode_target_path(obs_seq, cfg, agent_steps):
    """For envs whose target stream is a relative position (hunting), decode the
    target world path as agent_pos + (tgt_dx, tgt_dy, tgt_dz). Returns None if
    the env has no such stream."""
    if getattr(cfg, "TARGET_MOB", None) is None:
        return None
    p = cfg.PROPRIOCEPTION_SIZE
    steps = []
    for o, a in zip(obs_seq, agent_steps):
        dx, dy, dz = float(o[p + 0]), float(o[p + 1]), float(o[p + 2])
        steps.append({"x": a["x"] + dx, "y": a["y"] + dy, "z": a["z"] + dz})
    return steps


# ── Figures ──────────────────────────────────────────────────────────────────
def figure_overlay(imag, cfg, real_obs_seq=None, out_path=None, title="World-model imagination"):
    """3D overlay: imagined agent/target paths (dashed) vs real (solid)."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    imag_agent = decode_agent_path(imag["obs_seq"], cfg)
    render_imagined_trajectory(ax, imag_agent, color="#FF8800", label="imagined agent")
    imag_tgt = decode_target_path(imag["obs_seq"], cfg, imag_agent)
    if imag_tgt is not None:
        render_imagined_trajectory(ax, imag_tgt, color="#AA00FF", linestyle=":", label="imagined target")

    if real_obs_seq is not None:
        real_agent = decode_agent_path(real_obs_seq, cfg)
        render_trajectory(ax, real_agent, "alive")  # solid blue
        real_tgt = decode_target_path(real_obs_seq, cfg, real_agent)
        if real_tgt is not None:
            line, = ax.plot([s["x"] for s in real_tgt], [s["y"] for s in real_tgt],
                            [s["z"] for s in real_tgt], color="#1133AA", linewidth=1.5,
                            label="real target")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_path, "overlay")


def figure_metrics(imag, out_path=None):
    """Predicted reward / value / continuation traces over the horizon."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    H = len(imag["rewards"])
    t = np.arange(H)
    axes[0].plot(t, imag["rewards"], "-o", ms=3, color="#FF8800"); axes[0].set_ylabel("reward")
    axes[1].plot(np.arange(len(imag["values"])), imag["values"], "-o", ms=3, color="#4488FF")
    axes[1].set_ylabel("value")
    axes[2].plot(t, imag["conts"], "-o", ms=3, color="#00AA55"); axes[2].set_ylabel("P(cont)")
    axes[2].set_ylim(-0.05, 1.05); axes[2].set_xlabel("imagination step")
    axes[0].set_title("Imagined reward / value / continuation")
    fig.tight_layout()
    return _save(fig, out_path, "metrics")


def figure_voxel_montage(imag, cfg, out_path=None, y_slice=None):
    """Montage of the predicted occupancy grid (one horizontal slice) per step."""
    gx, gy, gz = cfg.GRID_X, cfg.GRID_Y, cfg.GRID_Z
    p = cfg.PROPRIOCEPTION_SIZE + cfg.GOAL_DELTA_SIZE
    if y_slice is None:
        y_slice = gy // 2

    obs_seq = imag["obs_seq"]
    n = len(obs_seq)
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < n:
            grid = obs_seq[i][p:p + gx * gy * gz].reshape(gx, gy, gz)
            ax.imshow(grid[:, y_slice, :], cmap="gray_r", vmin=0, vmax=1)
            ax.set_title("t={0}".format(i), fontsize=7)
    fig.suptitle("Imagined voxel occupancy (y-slice {0})".format(y_slice))
    fig.tight_layout()
    return _save(fig, out_path, "voxels")


# ── Ensemble rollout (uncertainty-aware) ──────────────────────────────────────
def imagine_ensemble(ensemble, model, start_obs, horizon, actions=None,
                     sample=False, device=None):
    """Roll every ensemble member from one start under a SHARED action sequence.

    Sharing the action sequence across members means the futures diverge because
    the members *disagree about the dynamics* (epistemic) — not because they took
    different actions. If `actions` is None the sequence is chosen once by the
    actor on the ensemble-mean rollout. With `sample=True` each member also draws
    its goal delta from its Gaussian, exposing aleatoric spread too.

    Returns dict:
        per_member (K, H+1, INPUT_SIZE)  imagined obs per member
        actions    (H,)                  the shared action sequence
        aleatoric  (H,)                  per-step aleatoric along the mean path
        epistemic  (H,)                  per-step epistemic along the mean path
    """
    device = device or next(model.parameters()).device
    members = list(ensemble.members)
    start = torch.as_tensor(np.asarray(start_obs, dtype=np.float32), device=device).unsqueeze(0)

    with torch.no_grad():
        if actions is None:
            actions = []
            s = start.clone()
            for _ in range(horizon):
                logits, _ = model(s)
                a = Categorical(logits=logits).sample()
                actions.append(int(a.item()))
                s, _, _ = ensemble.predict(s, a)
        actions = [int(a) for a in actions]

        per_member = []
        for m in members:
            s = start.clone()
            obs_seq = [s.squeeze(0).cpu().numpy()]
            for t in range(horizon):
                a = torch.tensor([actions[t]], device=device)
                s, _, _ = m.predict(s, a, sample=sample)
                obs_seq.append(s.squeeze(0).cpu().numpy())
            per_member.append(np.array(obs_seq, dtype=np.float32))

        ale, epi = [], []
        s = start.clone()
        for t in range(horizon):
            a = torch.tensor([actions[t]], device=device)
            _, _, _, a_u, e_u = ensemble.predict_with_uncertainty(s, a)
            ale.append(float(a_u.item())); epi.append(float(e_u.item()))
            s, _, _ = ensemble.predict(s, a)

    return {
        "per_member": np.stack(per_member),           # (K, H+1, D)
        "actions":    np.array(actions, dtype=np.int64),
        "aleatoric":  np.array(ale, dtype=np.float32),
        "epistemic":  np.array(epi, dtype=np.float32),
    }


def figure_dream_fan(ens_roll, cfg, out_path=None, title="Imagined futures — the ensemble hedge"):
    """Overlay every member's imagined TARGET (pig) path from one start.

    The paths fan out as the horizon grows and as the target becomes less
    predictable — a direct picture of the model hedging about the pig's future.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    per_member = ens_roll["per_member"]
    K = per_member.shape[0]
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, K))
    for k in range(K):
        agent_path = decode_agent_path(per_member[k], cfg)
        render_imagined_trajectory(ax, agent_path, color=colors[k], linestyle="-",
                                   label=("imagined agent" if k == 0 else None))
        tgt_path = decode_target_path(per_member[k], cfg, agent_path)
        if tgt_path is not None:
            render_imagined_trajectory(ax, tgt_path, color=colors[k], linestyle=":",
                                       label=("imagined pig (per member)" if k == 0 else None))

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_path, "fan")


def figure_uncertainty(ens_roll, out_path=None, title="Predictive uncertainty over the imagined horizon"):
    """Aleatoric vs epistemic (target-position) over the imagination horizon."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ale = ens_roll["aleatoric"]; epi = ens_roll["epistemic"]
    t = np.arange(len(ale))
    ax.plot(t, ale, "-o", ms=3, color="#CC4444", label="aleatoric (irreducible noise)")
    ax.plot(t, epi, "-o", ms=3, color="#4466CC", label="epistemic (model ignorance)")
    ax.plot(t, ale + epi, "--", color="#888888", label="total")
    ax.set_xlabel("imagination step"); ax.set_ylabel("variance (target position)")
    ax.set_title(title); ax.legend(fontsize=8); fig.tight_layout()
    return _save(fig, out_path, "uncertainty")


def figure_uncertainty_by_mode(rolls_by_mode, out_path=None,
                               title="Aleatoric uncertainty vs target predictability"):
    """The knob figure: aleatoric per TARGET_MODE (penned/wandering/fleeing).

    `rolls_by_mode` is {mode_name: ens_roll}. The claim under test is that
    aleatoric rises monotonically penned → wandering → fleeing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    modes = list(rolls_by_mode.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(modes)))
    for mode, c in zip(modes, colors):
        r = rolls_by_mode[mode]
        t = np.arange(len(r["aleatoric"]))
        axes[0].plot(t, r["aleatoric"], "-o", ms=3, color=c, label=mode)
        axes[1].plot(t, r["epistemic"], "-o", ms=3, color=c, label=mode)
    axes[0].set_title("aleatoric (should rise → fleeing)")
    axes[1].set_title("epistemic (should stay ~flat)")
    for ax in axes:
        ax.set_xlabel("imagination step"); ax.set_ylabel("variance"); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    return _save(fig, out_path, "knob")


def figure_counterfactual(ensemble, model, start_obs, action_seqs, cfg, horizon,
                          out_path=None, title="Counterfactual imagined branches"):
    """From ONE real start, imagine several candidate action sequences and overlay
    the branching agent+pig futures (ensemble-mean path per branch).

    `action_seqs` is {label: list[int] of length `horizon`}.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(action_seqs), 1)))
    for (label, seq), c in zip(action_seqs.items(), colors):
        roll = imagine_ensemble(ensemble, model, start_obs, horizon, actions=seq)
        mean_obs = roll["per_member"].mean(axis=0)
        agent_path = decode_agent_path(mean_obs, cfg)
        render_imagined_trajectory(ax, agent_path, color=c, linestyle="-", label=label)
        tgt_path = decode_target_path(mean_obs, cfg, agent_path)
        if tgt_path is not None:
            render_imagined_trajectory(ax, tgt_path, color=c, linestyle=":")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title + "  (solid=agent, dotted=pig)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_path, "counterfactual")


def _save(fig, out_path, tag):
    if out_path is None:
        return fig
    path = out_path if out_path.endswith(".png") else "{0}_{1}.png".format(out_path, tag)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


# ── Checkpoint loading ───────────────────────────────────────────────────────
def load_agent(checkpoint, cfg):
    """Build a Dreamer agent and load a checkpoint (ActorCritic + WorldModel)."""
    from models.actor_critic import ActorCritic
    from algos.dreamer import Dreamer
    model = ActorCritic(cfg)
    agent = Dreamer(model, cfg, n_envs=1)
    agent.load(checkpoint)
    agent.model.eval(); agent.world_model.eval()
    return agent


# ── CLI ──────────────────────────────────────────────────────────────────────
def _resolve_cfg(env_name):
    from training.train import ENV_REGISTRY
    return ENV_REGISTRY[env_name][1]()


def _default_counterfactuals(cfg, horizon):
    """A few illustrative held-action sequences to branch on."""
    names = getattr(cfg, "ACTIONS", None)
    candidates = [0, 4, 5, 6] if cfg.N_ACTIONS > 6 else list(range(min(3, cfg.N_ACTIONS)))
    seqs = {}
    for idx in candidates:
        label = names[idx][0] if names else "action_%d" % idx
        seqs[label] = [idx] * horizon
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="env name (resolves the config)")
    ap.add_argument("--checkpoint", required=True, help="dreamer checkpoint .pt")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--out", default="/tmp/dream", help="output path prefix")
    ap.add_argument("--mode", default="all",
                    choices=["all", "overlay", "metrics", "voxels",
                             "fan", "uncertainty", "counterfactual"],
                    help="which figure(s) to render "
                         "(fan/uncertainty/counterfactual need an ensemble checkpoint)")
    ap.add_argument("--sample", action="store_true",
                    help="fan: draw each member's goal delta from its Gaussian "
                         "(shows aleatoric spread too, not just epistemic)")
    ap.add_argument("--real-npz", default=None,
                    help="optional .npz with 'obs' (and 'actions') from a real "
                         "rollout for the open-loop overlay")
    args = ap.parse_args()

    cfg = _resolve_cfg(args.env)
    agent = load_agent(args.checkpoint, cfg)
    is_ensemble = hasattr(agent.world_model, "members")

    real_obs_seq, actions = None, None
    if args.real_npz:
        data = np.load(args.real_npz)
        real_obs_seq = data["obs"]
        actions = data["actions"] if "actions" in data else None
        start = real_obs_seq[0]
    else:
        start = np.zeros(cfg.INPUT_SIZE, dtype=np.float32)

    want = (lambda m: args.mode in ("all", m))

    # Single-model dream figures (work for ensemble too — predict returns the mean).
    if want("overlay") or want("metrics") or want("voxels"):
        imag = imagine(agent.world_model, agent.model, start, args.horizon, actions=actions)
        if want("overlay"):
            figure_overlay(imag, cfg, real_obs_seq=real_obs_seq, out_path=args.out)
        if want("metrics"):
            figure_metrics(imag, out_path=args.out)
        if want("voxels"):
            figure_voxel_montage(imag, cfg, out_path=args.out)

    # Ensemble-only uncertainty figures.
    if want("fan") or want("uncertainty") or want("counterfactual"):
        if not is_ensemble:
            print("[skip] fan/uncertainty/counterfactual need an ensemble "
                  "checkpoint (ENSEMBLE_SIZE>1); this one is a single model.")
        else:
            ens_roll = imagine_ensemble(agent.world_model, agent.model, start,
                                        args.horizon, actions=actions, sample=args.sample)
            if want("fan"):
                figure_dream_fan(ens_roll, cfg, out_path=args.out)
            if want("uncertainty"):
                figure_uncertainty(ens_roll, out_path=args.out)
            if want("counterfactual"):
                figure_counterfactual(agent.world_model, agent.model, start,
                                      _default_counterfactuals(cfg, args.horizon),
                                      cfg, args.horizon, out_path=args.out)


if __name__ == "__main__":
    main()
