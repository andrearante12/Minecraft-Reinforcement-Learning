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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="env name (resolves the config)")
    ap.add_argument("--checkpoint", required=True, help="dreamer checkpoint .pt")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--out", default="/tmp/dream", help="output path prefix")
    ap.add_argument("--real-npz", default=None,
                    help="optional .npz with 'obs' (and 'actions') from a real "
                         "rollout for the open-loop overlay")
    args = ap.parse_args()

    cfg = _resolve_cfg(args.env)
    agent = load_agent(args.checkpoint, cfg)

    real_obs_seq, actions = None, None
    if args.real_npz:
        data = np.load(args.real_npz)
        real_obs_seq = data["obs"]
        actions = data["actions"] if "actions" in data else None
        start = real_obs_seq[0]
    else:
        start = np.zeros(cfg.INPUT_SIZE, dtype=np.float32)

    imag = imagine(agent.world_model, agent.model, start, args.horizon, actions=actions)
    figure_overlay(imag, cfg, real_obs_seq=real_obs_seq, out_path=args.out)
    figure_metrics(imag, out_path=args.out)
    figure_voxel_montage(imag, cfg, out_path=args.out)


if __name__ == "__main__":
    main()
