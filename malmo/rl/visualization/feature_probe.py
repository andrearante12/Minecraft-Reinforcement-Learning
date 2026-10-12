"""
visualization/feature_probe.py
-------------------------------
Linear probe: can the FROZEN encoder see the rabbit?

The key Phase-A check before training the vision world model. On a recorded
frames demo set (utils/demo_io.py — the scripted teacher stores the rabbit's
privileged state next to every frame), fit ridge regressions from the frozen
tokens to:

  in_view  — rabbit inside the camera's horizontal field of view and within
             --max-dist (classification accuracy vs the majority-class rate)
  bearing  — sin/cos of the heading error, on in-view frames only (R²)
  distance — normalized distance, on in-view frames only (R²)

for three inputs: CLS token only, CLS + pooled patch grid, and a blind
proprio-only baseline. Splits are by EPISODE so neighbouring frames never
leak across train/test. If bearing R² from the grid stays near the CLS-only
number, the pooled grid is too coarse for a small rabbit: raise
VIS_POOL_GRID (try --grid 8).

Usage (conda: train_env):
    python malmo/rl/visualization/feature_probe.py --demos malmo/demos/hunting_vision_L1
    python malmo/rl/visualization/feature_probe.py --demos ... --grid 8 --encoder dinov2_vits14
"""

import argparse
import os
import sys

import numpy as np

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_ROOT)

from models.frozen_encoder import FrozenEncoder           # noqa: E402
from utils.demo_io import load_index, load_episode, episode_features   # noqa: E402

H_FOV_DEG = 70.0   # Minecraft default FOV; square frames → same horizontally


def ridge_fit(x, y, lam):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    a = xs.T @ xs + lam * np.eye(xs.shape[1])
    a[-1, -1] -= lam                                   # don't shrink the bias
    w = np.linalg.solve(a, xs.T @ y)
    return lambda z: np.hstack([(z - mu) / sd, np.ones((len(z), 1))]) @ w


def r2(pred, y):
    ss = ((y - y.mean(0)) ** 2).sum()
    return 1.0 - ((pred - y) ** 2).sum() / max(ss, 1e-9)


def collect(demo_dir, encoder, max_dist_norm):
    rows = []
    for ep_i, entry in enumerate(load_index(demo_dir)["episodes"]):
        ep = load_episode(demo_dir, entry, frames=False)
        if "privileged" not in ep:
            continue
        feats, _ = episode_features(demo_dir, entry, encoder)
        priv = ep["privileged"]
        in_view = ((np.abs(priv[:, 7]) * 180.0 < H_FOV_DEG / 2)
                   & (priv[:, 6] < max_dist_norm) & (priv[:, 11] > 0.5))
        rows.append({
            "ep": np.full(len(priv), ep_i),
            "cls": feats[:, 0].astype(np.float32),
            "grid": feats.reshape(len(feats), -1).astype(np.float32),
            "proprio": ep["vecs"].astype(np.float32),
            "in_view": in_view.astype(np.float32),
            "bearing": np.stack([np.sin(np.pi * priv[:, 7]), np.cos(np.pi * priv[:, 7])], 1),
            "dist": priv[:, 6:7],
        })
    if not rows:
        raise SystemExit("No episodes with privileged labels in {0} (record them with "
                         "utils/generate_hunting_demos.py --env hunting_vision)".format(demo_dir))
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demos", required=True, help="frames demo directory")
    ap.add_argument("--encoder", default="dinov2_vits14")
    ap.add_argument("--grid", type=int, default=4, help="patch pooling grid G (VIS_POOL_GRID)")
    ap.add_argument("--max-dist", type=float, default=0.5,
                    help="in_view requires dist_norm below this (default 0.5 = 15 blocks)")
    ap.add_argument("--lam", type=float, default=1000.0,
                    help="ridge strength (CLS+grid has ~6.5k dims: low values overfit)")
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    enc = FrozenEncoder(args.encoder, pool_grid=args.grid)
    d = collect(args.demos, enc, args.max_dist)
    eps = np.unique(d["ep"])
    rng = np.random.RandomState(0)
    test_eps = rng.choice(eps, size=max(1, int(round(len(eps) * args.test_frac))), replace=False)
    te = np.isin(d["ep"], test_eps)
    tr = ~te
    print("{0} frames from {1} episodes ({2} test episodes) | in-view rate {3:.2f}".format(
        len(te), len(eps), len(test_eps), d["in_view"].mean()))

    majority = max(d["in_view"][te].mean(), 1 - d["in_view"][te].mean())
    print("\n{0:<10} {1:>14} {2:>12} {3:>12}".format("input", "in_view acc", "bearing R2", "dist R2"))
    for name in ("proprio", "cls", "grid"):
        x = d[name]
        f = ridge_fit(x[tr], d["in_view"][tr, None], args.lam)
        acc = ((f(x[te])[:, 0] > 0.5) == (d["in_view"][te] > 0.5)).mean()
        vis_tr, vis_te = tr & (d["in_view"] > 0.5), te & (d["in_view"] > 0.5)
        if vis_tr.sum() > 10 and vis_te.sum() > 5:
            b = r2(ridge_fit(x[vis_tr], d["bearing"][vis_tr], args.lam)(x[vis_te]), d["bearing"][vis_te])
            dd = r2(ridge_fit(x[vis_tr], d["dist"][vis_tr], args.lam)(x[vis_te]), d["dist"][vis_te])
        else:
            b = dd = float("nan")
        label = name if name != "grid" else "cls+{0}x{0}".format(args.grid)
        print("{0:<10} {1:>14.3f} {2:>12.3f} {3:>12.3f}".format(label, acc, b, dd))
    print("\n(majority-class in_view accuracy: {0:.3f}; proprio is the blind baseline)".format(majority))


if __name__ == "__main__":
    main()
