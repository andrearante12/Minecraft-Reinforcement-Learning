"""
evaluation/live_imagination.py
------------------------------
Watch the world model's imagination INSIDE Minecraft, and score it against what
actually happens (live Experiment 2).

Each real step:
  1. imagine H steps ahead from the current real obs (actor acts greedily inside
     the frozen model; ensemble members replay the same actions → the fan)
  2. draw that future as /particle commands in the game (visualization/ingame_overlay.py)
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
    imagine_open_loop, divergence_curve, divergence_horizon)
from visualization.ingame_overlay import build_commands, probe_commands, Throttle


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


def run(client, agent, cfg, args):
    scorer = Scorer(agent, cfg, args.horizon)
    throttle = Throttle(args.max_cmds_per_sec)
    n_sent = 0
    for ep in range(args.episodes):
        obs = client.reset()
        obs_hist, act_hist, done, info = [obs], [], False, {}
        while not done:
            seq, acts, members = plan(agent, obs, args.horizon)
            if not args.no_draw:
                cmds = build_commands(seq, cfg, members_obs=members, max_cmds=args.max_cmds)
                k = throttle.take(len(cmds))
                if k:
                    n_sent += client.chat(cmds[:k])
            a = agent.select_action(obs, greedy=True)
            obs, r, done, info = client.step(a)
            obs_hist.append(obs); act_hist.append(a)
        scorer.score_episode(obs_hist, act_hist)
        print("episode {}: {} steps, outcome={}".format(ep + 1, len(act_hist), info.get("outcome")))
    out = scorer.summary()
    out["commands_sent"] = n_sent
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
    ap.add_argument("--no-draw", action="store_true", help="score only; draw nothing in game")
    ap.add_argument("--probe", action="store_true", help="draw one marker per particle type and exit")
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
        print("sent", client.chat(cmds), "markers 3 blocks ahead of spawn, left→right:")
        from visualization.ingame_overlay import STYLE
        print("  " + "  ".join(STYLE.values()))
        print("If a marker is missing, that particle name isn't valid in this Minecraft "
              "version — edit STYLE in visualization/ingame_overlay.py.")
        return

    agent = load_agent(cfg, args.checkpoint)
    t0 = time.time()
    summary = run(client, agent, cfg, args)
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
