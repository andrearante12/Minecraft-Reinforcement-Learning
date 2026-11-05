"""
evaluation/evaluate_hunting.py
------------------------------
Evaluate a trained hunting_vision agent (ppo_vision or dreamer_video) over TCP,
optionally with Dreamer's lookahead planner (our addition, evaluation only).

The biome is chosen on the env-server side, e.g. the held-out forest:
    MALMO_WORLD_SEEDS=forest python envs/env_server.py --env hunting_vision --port 9999 --malmo-port 10010

Then (conda: train_env):
    python evaluation/evaluate_hunting.py --algo dreamer_video \\
        --checkpoint checkpoints/<run>_final.pt --episodes 20 --port 9999 \\
        [--plan --plan-candidates 4 --plan-samples 8 --plan-horizon 8 --plan-score mean] \\
        [--csv logs/eval_dreamer_plan_forest.csv]

Per episode: outcome, reward, steps, kills, world seed / biome, and the mean
wall-clock time to choose an action (the planner's extra cost).
"""

import argparse
import csv
import os
import sys
import time

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

import torch

from training.configs.hunting_vision_cfg import HuntingVisionCFG, BIOME_SEEDS
from models.latent_actor_critic import LatentActorCritic
from models.vision_actor_critic import VisionActorCritic
from algos.dreamer_video import DreamerVideo
from algos.ppo_vision import PPOVision
from envs.env_client import EnvClient

HAZARDS = {"fell", "stuck_in_pit", "drowned", "lava", "burned", "suffocated", "hazard"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo_vision", "dreamer_video"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--fair", action="store_true",
                   help="dreamer_video: build the model as trained with train.py --fair (no privileged head)")
    p.add_argument("--plan", action="store_true", help="dreamer_video: use the lookahead planner")
    p.add_argument("--plan-candidates", type=int, default=None)
    p.add_argument("--plan-samples", type=int, default=None)
    p.add_argument("--plan-horizon", type=int, default=None)
    p.add_argument("--plan-score", choices=["mean", "worst"], default=None)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy")
    p.add_argument("--csv", default=None, help="write per-episode rows here")
    p.add_argument("--cfg", action="append", default=[], metavar="KEY=VALUE",
                   help="config overrides the checkpoint was trained with (e.g. VWM_REWARD_ENSEMBLE=5)")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = HuntingVisionCFG
    if a.algo == "dreamer_video":
        if a.fair:
            from training.train import apply_fair_settings
            apply_fair_settings(cfg, a.algo)
        cfg.PLAN_AT_ACT = bool(a.plan)
        for flag, attr in [("plan_candidates", "PLAN_CANDIDATES"), ("plan_samples", "PLAN_SAMPLES"),
                           ("plan_horizon", "PLAN_HORIZON"), ("plan_score", "PLAN_SCORE")]:
            if getattr(a, flag) is not None:
                setattr(cfg, attr, getattr(a, flag))
    elif a.plan:
        raise SystemExit("--plan needs --algo dreamer_video (PPO has no world model)")

    from training.train import apply_cfg_overrides
    apply_cfg_overrides(cfg, a.cfg)
    names = {str(v): k for k, v in BIOME_SEEDS.items()}
    env = EnvClient(cfg.INPUT_SIZE, port=a.port, video=True,
                    frame_shape=(cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS))
    if a.algo == "ppo_vision":
        agent = PPOVision(VisionActorCritic(cfg), cfg)
    else:
        agent = DreamerVideo(LatentActorCritic(cfg), cfg)
    agent.load(a.checkpoint)

    label = a.algo + (" + planner (C={0}, K={1}, H={2}, {3})".format(
        cfg.PLAN_CANDIDATES, cfg.PLAN_SAMPLES, cfg.PLAN_HORIZON, cfg.PLAN_SCORE) if a.plan else "")
    print("=" * 60)
    print("Hunting evaluation:", label)
    print("Checkpoint:", a.checkpoint, "| episodes:", a.episodes, "| port:", a.port)
    print("=" * 60)

    rows = []
    for ep in range(1, a.episodes + 1):
        if hasattr(agent, "reset_eval_state"):
            agent.reset_eval_state()
        obs, done = env.reset(), False
        total, steps, act_time, info = 0.0, 0, 0.0, {}
        while not done:
            t0 = time.time()
            with torch.no_grad():
                action = agent.select_action(obs, greedy=not a.stochastic)
            act_time += time.time() - t0
            obs, reward, done, info = env.step(action)
            total += reward
            steps += 1
        seed = str(info.get("world_seed", ""))
        row = {"episode": ep, "outcome": info.get("outcome", "?"), "reward": round(total, 3),
               "steps": steps, "kills": info.get("kills", 0), "world_seed": seed,
               "biome": names.get(seed, seed), "ms_per_action": round(1000 * act_time / max(steps, 1), 2)}
        rows.append(row)
        print("  Ep {episode:>3} | {biome:<7} | steps:{steps:>4} | reward:{reward:>8.2f} | "
              "kills:{kills} | {outcome:<12} | {ms_per_action:.1f} ms/action".format(**row))

    n = float(len(rows))
    print("\n" + "=" * 60)
    print("Summary ({0} episodes): {1}".format(len(rows), label))
    print("  killed:        {0:.0f}%".format(100 * sum(r["outcome"] == "killed" for r in rows) / n))
    print("  hazard deaths: {0:.0f}%".format(100 * sum(r["outcome"] in HAZARDS for r in rows) / n))
    print("  timeouts:      {0:.0f}%".format(100 * sum(r["outcome"] == "timeout" for r in rows) / n))
    print("  mean kills:    {0:.2f}".format(sum(r["kills"] for r in rows) / n))
    print("  mean reward:   {0:.2f}".format(sum(r["reward"] for r in rows) / n))
    print("  mean steps:    {0:.1f}".format(sum(r["steps"] for r in rows) / n))
    print("  ms per action: {0:.1f}".format(sum(r["ms_per_action"] for r in rows) / n))

    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["label"])
            w.writeheader()
            for r in rows:
                w.writerow(dict(r, label=label))
        print("wrote", a.csv)
    env.close()


if __name__ == "__main__":
    main()
