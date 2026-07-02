"""
utils/generate_hunting_demos.py
--------------------------------
Scripted demo generator for the hunting task.

Drives the env with a heuristic hunter that navigates toward the pig and
attacks when aligned and in range. Saves full (obs, action, reward, next_obs,
done) transitions so the Dreamer demo buffer can train both the world model
and the BC anchor.

Usage:
    conda activate train_env
    python malmo/rl/utils/generate_hunting_demos.py --port 9999 --episodes 30
    python malmo/rl/utils/generate_hunting_demos.py --port 9999 --episodes 50 --output demos/hunting.json

Controls are automated — just run and watch. The scripted policy is:
  1. If not visible → spin slowly to find the pig.
  2. If heading error > ALIGN_THRESH → turn toward pig.
  3. If too far away → move forward.
  4. If in range and roughly aligned → attack.
  5. Else → move forward + turn simultaneously (two-step).
"""

import sys
import os
import json
import argparse
import time
import functools
import math

print = functools.partial(print, flush=True)

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from envs.env_client import EnvClient
from training.configs.hunting_cfg import HuntingCFG


# ── Action indices (must match HuntingCFG.HUNTING_ACTIONS) ──────────────────
ACT_FORWARD  = 0
ACT_BACKWARD = 1
ACT_LEFT     = 2
ACT_RIGHT    = 3
ACT_SPRINT   = 4
ACT_TURN_L   = 5
ACT_TURN_R   = 6
ACT_LOOK_U   = 7
ACT_LOOK_D   = 8
ACT_ATTACK   = 9
ACT_NOOP     = 10

# ── Observation indices (HuntingCFG layout) ──────────────────────────────────
# [17] tgt_dist_norm  [18] heading_error_norm  [20] in_range  [22] tgt_visible
OBS_TGT_DIST    = 17
OBS_HEADING_ERR = 18
OBS_LOS_HIT     = 19
OBS_IN_RANGE    = 20
OBS_TGT_VISIBLE = 22


def scripted_action(obs, cfg):
    """Heuristic hunting policy → returns an action index."""
    visible    = obs[OBS_TGT_VISIBLE] > 0.5
    heading    = obs[OBS_HEADING_ERR]   # normalized -1..1
    in_range   = obs[OBS_IN_RANGE] > 0.5
    dist       = obs[OBS_TGT_DIST]     # normalized; 1.0 = DIST_SCALE away

    ALIGN_THRESH  = 0.12   # ~22° heading error — start attacking
    TURN_THRESH   = 0.07   # fine-tune turn vs move decision
    APPROACH_DIST = 0.15   # dist_norm; closer → prefer attack over approach

    if not visible:
        return ACT_TURN_R   # slow spin to find the pig

    # Attack if aligned enough and in range
    if in_range and abs(heading) < ALIGN_THRESH:
        return ACT_ATTACK

    # Correct yaw first if significantly off
    if abs(heading) > TURN_THRESH:
        return ACT_TURN_R if heading > 0 else ACT_TURN_L

    # Move forward if far away (and roughly aligned)
    if dist > APPROACH_DIST:
        return ACT_SPRINT if dist > 0.4 else ACT_FORWARD

    # Close but not yet aligned — fine-tune turn
    if heading > 0:
        return ACT_TURN_R
    elif heading < 0:
        return ACT_TURN_L

    return ACT_ATTACK


def parse_args():
    parser = argparse.ArgumentParser(description="Generate scripted hunting demos")
    parser.add_argument("--port",     type=int, default=9999,
                        help="Env server TCP port (default: 9999)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes to record (default: 30)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Output JSON path (default: demos/hunting.json)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (default: cfg.MAX_STEPS)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = HuntingCFG()
    max_steps = args.max_steps or cfg.MAX_STEPS

    output_path = args.output or os.path.join(
        os.path.dirname(PARKOUR_ROOT), "demos", "hunting.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load existing file to append
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        print("Appending to {0} existing episodes in {1}".format(
            len(data["episodes"]), output_path))
    else:
        data = {"env": "hunting", "episodes": []}

    env = EnvClient(cfg.INPUT_SIZE, port=args.port)

    print("=" * 60)
    print("Scripted Hunting Demo Generator")
    print("=" * 60)
    print("Port:     ", args.port)
    print("Episodes: ", args.episodes)
    print("Max steps:", max_steps)
    print("Output:   ", output_path)
    print("=" * 60)

    kills = 0
    try:
        for ep_idx in range(args.episodes):
            print("\nEpisode {0}/{1}...".format(ep_idx + 1, args.episodes))
            obs = env.reset()
            steps = []
            total_reward = 0.0
            done = False
            outcome = "timeout"

            for step_i in range(max_steps):
                action = scripted_action(obs, cfg)
                obs_prev = obs.copy()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps.append({
                    "obs":      obs_prev.tolist(),
                    "action":   int(action),
                    "reward":   float(reward),
                    "next_obs": obs.tolist(),
                    "done":     bool(done),
                })
                if done:
                    outcome = info.get("outcome", "unknown")
                    break

            if outcome == "killed":
                kills += 1
            print("  steps:{0}  reward:{1:.1f}  outcome:{2}".format(
                len(steps), total_reward, outcome))

            if steps:
                data["episodes"].append({"outcome": outcome, "steps": steps})
                with open(output_path, "w") as f:
                    json.dump(data, f)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()

    total_eps   = len(data["episodes"])
    total_steps = sum(len(ep["steps"]) for ep in data["episodes"])
    print("\nDone.")
    print("  Episodes recorded this run: {0}  (kills: {1}/{0})".format(
        args.episodes, kills))
    print("  Total in file: {0} eps, {1} steps".format(total_eps, total_steps))
    print("  Saved to: {0}".format(output_path))


if __name__ == "__main__":
    main()
