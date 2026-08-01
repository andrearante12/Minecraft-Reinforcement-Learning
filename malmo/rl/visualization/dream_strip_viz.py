"""
visualization/dream_strip_viz.py
----------------------------------
Film-strip visualization for the video world model (models/video_world_model.py,
algos/dreamer_video.py) — the pixel-space analogue of visualization/imagination_viz.py,
which is left untouched (it only knows the vector world model).

Three rows, one column per timestep:
  1. real context frames      — a short real sequence fed to the RSSM posterior
  2. posterior reconstructions — decode(posterior feat) at each context step:
                                  "what the model believes it just saw"
  3. imagined continuation     — prior-only rollout from the last context state,
                                  decoded to frames, with predicted reward/continuation
                                  printed under each column

With no --context given, row 1/2 are skipped and the dream starts from the
RSSM's zero initial state (pure imagination, no real data needed).

CLI (run in train_env):
    python visualization/dream_strip_viz.py --env hunting_video \
        --checkpoint checkpoints/dreamer_video_hunting_video_ep100.pt --out /tmp/dreamstrip
    python visualization/dream_strip_viz.py --env hunting_video \
        --checkpoint ... --context /tmp/context_seq.npz --out /tmp/dreamstrip
        # context_seq.npz: "frames" (L,H,W,C) uint8, "vecs" (L,D) f32, optional "actions" (L,) int64
"""

import os
import sys
import argparse
import numpy as np

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Checkpoint loading ───────────────────────────────────────────────────────
def load_models(checkpoint, cfg, device=None):
    """Load VideoWorldModel + LatentActorCritic from a dreamer_video checkpoint.
    Does not build a full DreamerVideo agent (no optimizer state needed for viz)."""
    from models.video_world_model import VideoWorldModel
    from models.latent_actor_critic import LatentActorCritic

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "video_world_model_state" not in ckpt:
        raise ValueError(
            "'{0}' has no video_world_model_state key — this is not a dreamer_video "
            "checkpoint (vector dreamer checkpoints use 'world_model_state' and are "
            "not compatible with this viz; use visualization/imagination_viz.py instead).".format(checkpoint))

    world_model = VideoWorldModel(cfg).to(device)
    world_model.load_state_dict(ckpt["video_world_model_state"])
    model = LatentActorCritic(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    world_model.eval()
    model.eval()
    return world_model, model, device


# ── Rollouts ─────────────────────────────────────────────────────────────────
def posterior_pass(world_model, frames, vecs, actions, device):
    """Run the RSSM posterior over a real context sequence.

    frames: (L,H,W,C) uint8   vecs: (L,D) f32   actions: (L,) int64 or None
    (if actions is None, zeros are used — acceptable for a context whose sole
    purpose is seeding the latent state, not for training).
    Returns (posts, recon_frames [0,1]-space (L,H,W,C), last_state).
    """
    L = frames.shape[0]
    n_actions = world_model.n_actions
    frames_t = torch.as_tensor(frames, device=device).unsqueeze(0)
    vecs_t = torch.as_tensor(vecs, dtype=torch.float32, device=device).unsqueeze(0)
    if actions is None:
        actions = np.zeros(L, dtype=np.int64)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(0)
    actions_onehot = F.one_hot(actions_t, n_actions).float()
    prev_actions = world_model._shift_actions(actions_onehot)

    with torch.no_grad():
        embed, _ = world_model._embed(frames_t, vecs_t)
        posts, _ = world_model.rssm.observe(embed, prev_actions)
        feat = world_model.rssm.get_feat(posts)
        recon_frame, _ = world_model.decode(feat)
    recon_np = (recon_frame.squeeze(0) + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    last_state = {k: v[:, -1] for k, v in posts.items()}
    return posts, recon_np, last_state


def dream_continuation(world_model, model, start_state, horizon, actions=None, device=None):
    """Prior-only rollout from start_state. If `actions` (length-horizon int
    sequence) is given, replay open-loop; otherwise the actor samples ("dream").

    Returns dict: frames [0,1]-space (H,H_img,W_img,C), rewards (H,), conts (H,).
    """
    device = device or next(model.parameters()).device
    rssm = world_model.rssm
    n_actions = world_model.n_actions
    state = {k: v.clone() for k, v in start_state.items()}

    frames, rewards, conts = [], [], []
    with torch.no_grad():
        for t in range(horizon):
            feat = rssm.get_feat(state)
            if actions is not None:
                a = torch.tensor([int(actions[t])], device=device)
            else:
                logits, _ = model(feat)
                a = Categorical(logits=logits).sample()
            a_onehot = F.one_hot(a, n_actions).float()
            r, cont = world_model.predict_reward_cont(feat, a_onehot)
            state = rssm.img_step(state, a_onehot)
            recon_frame, _ = world_model.decode(rssm.get_feat(state))
            frames.append((recon_frame.squeeze(0) + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
            rewards.append(float(r.item()))
            conts.append(float(cont.item()))

    return {
        "frames": np.stack(frames),
        "rewards": np.array(rewards, dtype=np.float32),
        "conts": np.array(conts, dtype=np.float32),
    }


# ── Figure ───────────────────────────────────────────────────────────────────
def figure_dream_strip(real_frames, recon_frames, dream, out_path=None):
    """real_frames/recon_frames: (Lc,H,W,C) or None. dream: dict from dream_continuation."""
    n_ctx = 0 if real_frames is None else real_frames.shape[0]
    n_dream = dream["frames"].shape[0]
    n_cols = max(n_ctx, n_dream)
    n_rows = 3 if n_ctx > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows + 0.6), squeeze=False)

    if n_ctx > 0:
        for c in range(n_cols):
            axes[0, c].axis("off")
            axes[1, c].axis("off")
            if c < n_ctx:
                axes[0, c].imshow(real_frames[c].astype(np.float32) / 255.0)
                axes[1, c].imshow(recon_frames[c])
        axes[0, 0].set_ylabel("real", fontsize=9)
        axes[1, 0].set_ylabel("posterior recon", fontsize=9)
        dream_row = 2
    else:
        dream_row = 0

    for c in range(n_cols):
        axes[dream_row, c].axis("off")
        if c < n_dream:
            axes[dream_row, c].imshow(dream["frames"][c])
            axes[dream_row, c].set_title(
                "r={0:.2f}\nc={1:.2f}".format(dream["rewards"][c], dream["conts"][c]),
                fontsize=7)

    fig.suptitle("Video world model dream strip"
                  + ("" if n_ctx > 0 else "  (pure imagination, no context)"), fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path, "dreamstrip")


def _save(fig, out_path, tag):
    if out_path is None:
        return fig
    path = out_path if out_path.endswith(".png") else "{0}_{1}.png".format(out_path, tag)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────
def _resolve_cfg(env_name):
    from training.train import ENV_REGISTRY
    return ENV_REGISTRY[env_name][1]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="env name (resolves the config, e.g. hunting_video)")
    ap.add_argument("--checkpoint", required=True, help="dreamer_video checkpoint .pt")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default="/tmp/dream", help="output path prefix")
    ap.add_argument("--context", default=None,
                    help="optional .npz with 'frames' (L,H,W,C) uint8, 'vecs' (L,D) f32, "
                         "optional 'actions' (L,) int64 — a short real sequence to seed the "
                         "RSSM posterior before dreaming. Without this, dreaming starts from "
                         "the RSSM's zero initial state (pure imagination).")
    args = ap.parse_args()

    cfg = _resolve_cfg(args.env)
    world_model, model, device = load_models(args.checkpoint, cfg)

    if args.context:
        data = np.load(args.context)
        frames = data["frames"]
        vecs = data["vecs"]
        ctx_actions = data["actions"] if "actions" in data else None
        _, recon_np, last_state = posterior_pass(world_model, frames, vecs, ctx_actions, device)
    else:
        frames, recon_np = None, None
        last_state = world_model.rssm.initial(1, device)

    dream = dream_continuation(world_model, model, last_state, args.horizon, device=device)
    figure_dream_strip(frames, recon_np, dream, out_path=args.out)


if __name__ == "__main__":
    main()
