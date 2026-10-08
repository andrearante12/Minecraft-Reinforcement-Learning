"""
evaluation/live_imagination.py
------------------------------
Watch the world model's imagination INSIDE Minecraft, and score it against what
actually happens (live Experiment 2).

Each real step:
  1. imagine H steps ahead from the current real obs (actor acts greedily inside
     the frozen model; ensemble members replay the same actions → the fan)
  2. publish it to a live top-down view at http://localhost:8770 (run it beside Minecraft);
     optionally (--particles) also draw it as /particle commands in the game
  3. execute the policy's real action
Afterwards, for every window of H real steps, the model is re-run OPEN-LOOP on the
actions that were actually taken and compared with the real observations →
error-vs-horizon and divergence horizon (evaluation/imagination_vs_reality.py).

Live (3 terminals; note --viz on the env server):
    T1  Minecraft client
    T2  python malmo/rl/envs/env_server.py --env hunting --port 9999 --malmo-port 10000 --viz
    T3  python malmo/rl/evaluation/live_imagination.py --checkpoint <dreamer_ckpt.pt> --port 9999
        add --probe first to check which particle names your Minecraft build supports

Offline (no Malmo; stand-in env, prints the commands instead of sending them):
    python malmo/rl/evaluation/live_imagination.py --dry-run [--checkpoint ...]
"""

import os
import sys
import json
import time
import argparse
import numpy as np

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_ROOT)

import torch

from training.configs.hunting_cfg import HuntingCFG
from models.actor_critic import ActorCritic
from algos.dreamer import Dreamer
from evaluation.imagination_vs_reality import (
    imagine_open_loop, divergence_curve, divergence_horizon, decode_positions)
from visualization.ingame_overlay import build_commands, probe_commands, Throttle
from visualization.live_view_server import LiveState, PlanErrorTracker, serve


# ── planning ─────────────────────────────────────────────────────────────────
def plan(agent, obs, horizon):
    """Greedy actor inside the frozen model. Returns (obs_seq (H+1,D), actions (H,),
    members (K,H+1,D) or None)."""
    dev = agent.device
    wm, pi = agent.world_model, agent.model
    s = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=dev).unsqueeze(0)
    seq, acts = [s.squeeze(0).cpu().numpy()], []
    with torch.no_grad():
        for _ in range(horizon):
            logits, _ = pi(s)
            a = logits.argmax(dim=-1)
            s, _, _ = wm.predict(s, a)
            acts.append(int(a.item()))
            seq.append(s.squeeze(0).cpu().numpy())
        members = None
        if getattr(agent, "is_ensemble", False):
            members = []
            for m in wm.members:
                x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=dev).unsqueeze(0)
                ms = [x.squeeze(0).cpu().numpy()]
                for a in acts:
                    x, _, _ = m.predict(x, torch.tensor([a], device=dev))
                    ms.append(x.squeeze(0).cpu().numpy())
                members.append(np.array(ms, dtype=np.float32))
            members = np.stack(members)
    return np.array(seq, dtype=np.float32), np.array(acts), members


def uncertainty_along(agent, seq, acts):
    """(aleatoric (H,), epistemic (H,)) along a plan; zeros for a single model."""
    if not getattr(agent, "is_ensemble", False):
        z = np.zeros(len(acts)); return z, z
    dev = agent.device
    with torch.no_grad():
        s = torch.as_tensor(seq[:-1], dtype=torch.float32, device=dev)
        a = torch.as_tensor(acts, dtype=torch.int64, device=dev)
        _, _, _, ale, epi = agent.world_model.predict_with_uncertainty(s, a)
    return ale.cpu().numpy(), epi.cpu().numpy()


def _xz(p):
    return [[round(float(v[0]), 3), round(float(v[2]), 3)] for v in p]


# ── scoring (live Experiment 2) ──────────────────────────────────────────────
class Scorer:
    def __init__(self, agent, cfg, horizon, threshold=1.0):
        self.agent, self.cfg, self.H, self.thr = agent, cfg, horizon, threshold
        self.curves, self.horizons = [], []

    def score_episode(self, obs_hist, act_hist):
        """Slide a window of H steps over the finished episode."""
        H = self.H
        for t0 in range(0, len(act_hist) - H + 1):
            real = np.array(obs_hist[t0:t0 + H + 1])
            im = imagine_open_loop(self.agent.world_model, real[0], act_hist[t0:t0 + H])
            err = divergence_curve(im["obs_seq"], real, self.cfg)["err"]
            self.curves.append(err)
            self.horizons.append(divergence_horizon(err, self.thr))

    def summary(self):
        if not self.curves:
            return {"windows": 0}
        return {"windows": len(self.curves), "horizon": self.H, "threshold": self.thr,
                "median_divergence_horizon": float(np.median(self.horizons)),
                "mean_err_by_step": np.mean(self.curves, 0).round(3).tolist()}


# ── clients ──────────────────────────────────────────────────────────────────
class DryRunClient:
    """Stand-in for EnvClient: SimHunting + a chat() that records commands."""

    def __init__(self, cfg, mode, seed=0):
        from utils.sim_hunting import SimHunting
        self.env = SimHunting(cfg, mode, seed=seed)
        self.sent = []

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)

    def chat(self, commands):
        self.sent.append(list(commands))
        return len(commands)

    def close(self):
        pass


def run(client, agent, cfg, args, state=None):
    scorer = Scorer(agent, cfg, args.horizon)
    throttle = Throttle(args.max_cmds_per_sec)
    tracker = PlanErrorTracker(args.horizon)
    names = [x[0] for x in cfg.ACTIONS]
    p = cfg.PROPRIOCEPTION_SIZE
    n_sent, last_outcome, ghost = 0, None, None
    for ep in range(args.episodes):
        obs = client.reset()
        tracker.reset_episode()
        trail_a, trail_p, ghost = [], [], None
        obs_hist, act_hist, done, info = [obs], [], False, {}
        while not done:
            seq, acts, members = plan(agent, obs, args.horizon)
            ag, pg = decode_positions(seq, cfg)
            tracker.add_plan(ag, pg)
            ra, rp = decode_positions(np.asarray(obs)[None], cfg)
            trail_a.append(ra[0]); trail_p.append(rp[0])
            a = agent.select_action(obs, greedy=True)

            if args.particles:
                cmds = build_commands(seq, cfg, members_obs=members, max_cmds=args.max_cmds)
                k = throttle.take(len(cmds))
                if k:
                    n_sent += client.chat(cmds[:k])
            if state is not None:
                ale, epi = uncertainty_along(agent, seq, acts)
                mem = [] if members is None else [_xz(decode_positions(m, cfg)[1]) for m in members]
                state.publish({
                    "episode": ep + 1, "step": len(act_hist), "H": args.horizon,
                    "arena": [cfg.ARENA_MIN, cfg.ARENA_MAX], "last_outcome": last_outcome,
                    "action": names[a],
                    "real": {"agent": _xz(ra)[0], "pig": _xz(rp)[0], "yaw": float(obs[1]) * 180.0,
                             "trail_agent": _xz(trail_a), "trail_pig": _xz(trail_p)},
                    "plan": {"agent": _xz(ag), "pig": _xz(pg), "members": mem},
                    "ghost": None if ghost is None else {"agent": [ghost["agent"][0], ghost["agent"][2]],
                                                         "pig": None if ghost["pig"] is None else
                                                         [ghost["pig"][0], ghost["pig"][2]]},
                    "obs": {"dist": float(obs[p + 6]) * cfg.DIST_SCALE, "herr": float(obs[p + 7]),
                            "in_range": bool(obs[p + 9] > 0.5), "life": float(obs[p + 10])},
                    "ale": [round(float(x), 5) for x in ale], "epi": [round(float(x), 5) for x in epi],
                    "err_by_k": tracker.mean_by_k(),
                })
            obs, r, done, info = client.step(a)
            if args.step_delay:
                time.sleep(args.step_delay)
            ra2, rp2 = decode_positions(np.asarray(obs)[None], cfg)
            ghost = tracker.update(ra2[0], rp2[0]) or ghost
            obs_hist.append(obs); act_hist.append(a)
        last_outcome = info.get("outcome")
        scorer.score_episode(obs_hist, act_hist)
        print("episode {}: {} steps, outcome={}".format(ep + 1, len(act_hist), last_outcome))
    out = scorer.summary()
    out["commands_sent"] = n_sent
    out["online_plan_error_by_k"] = tracker.mean_by_k()
    return out


def load_agent(cfg, ckpt):
    agent = Dreamer(ActorCritic(cfg), cfg)
    if ckpt:
        agent.load(ckpt)
    else:
        print("WARNING: no --checkpoint — using an UNTRAINED agent (plumbing check only)")
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Dreamer checkpoint (policy + world model)")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--target-mode", default=None, choices=["penned", "wandering", "fleeing"])
    ap.add_argument("--max-cmds", type=int, default=16, help="particle commands per real step")
    ap.add_argument("--max-cmds-per-sec", type=float, default=40.0)
    ap.add_argument("--particles", action="store_true",
                    help="ALSO draw the imagined future as in-game particles (needs env_server --viz)")
    ap.add_argument("--view-port", type=int, default=8770,
                    help="serve the live top-down view at http://localhost:PORT (0 = off)")
    ap.add_argument("--step-delay", type=float, default=None,
                    help="extra seconds per step (dry-run defaults to 0.12 so the view is watchable)")
    ap.add_argument("--probe", action="store_true", help="draw one marker per particle type and exit")
    ap.add_argument("--probe-seconds", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true", help="stand-in env, no Malmo")
    ap.add_argument("--out", default=None, help="write summary JSON here")
    args = ap.parse_args()

    cfg = HuntingCFG
    if args.target_mode:
        class cfg(HuntingCFG):  # noqa: N801
            TARGET_MODE = args.target_mode

    if args.dry_run:
        client = DryRunClient(cfg, cfg.TARGET_MODE)
    else:
        from envs.env_client import EnvClient
        client = EnvClient(cfg.INPUT_SIZE, port=args.port)

    if args.probe:
        obs = client.reset()
        cmds = probe_commands(cfg.SPAWN)
        t_end = time.time() + args.probe_seconds
        sent = 0
        while time.time() < t_end:          # particles last ~1 s → keep re-drawing
            sent += client.chat(cmds)
            time.sleep(0.25)
        print("sent", sent, "commands; markers 3 blocks ahead of spawn, left→right:")
        from visualization.ingame_overlay import STYLE
        print("  " + "  ".join(STYLE.values()))
        print("If a marker is missing, that particle name isn't valid in this Minecraft "
              "version — edit STYLE in visualization/ingame_overlay.py.")
        return

    agent = load_agent(cfg, args.checkpoint)
    if args.step_delay is None:
        args.step_delay = 0.12 if args.dry_run else 0.0
    state = None
    if args.view_port:
        state = LiveState()
        serve(state, args.view_port)
        print("Live view: http://localhost:{}".format(args.view_port))
    t0 = time.time()
    summary = run(client, agent, cfg, args, state)
    summary["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
    if args.dry_run and client.sent:
        print("sample commands for one step:")
        for c in client.sent[len(client.sent) // 2][:6]:
            print("  ", c)
    client.close()


if __name__ == "__main__":
    main()
