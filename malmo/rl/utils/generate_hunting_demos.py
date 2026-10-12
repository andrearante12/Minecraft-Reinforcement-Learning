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

    # Vision task (pixels + proprio): a PRIVILEGED teacher — it acts on the
    # rabbit's true position from info["privileged"], while the saved demos hold
    # what the student sees (camera frame + proprio). Writes a frames demo
    # directory (utils/demo_io.py), default malmo/demos/hunting_vision_L<level>/.
    python malmo/rl/utils/generate_hunting_demos.py --env hunting_vision --port 9999 --episodes 50

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
from training.configs.hunting_wild_cfg import HuntingWildCFG
from training.configs.hunting_vision_cfg import HuntingVisionCFG

ENV_CONFIGS = {
    "hunting":        HuntingCFG,
    "hunting_wild":   HuntingWildCFG,
    "hunting_vision": HuntingVisionCFG,
}


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
ACT_JUMP_FWD = 11   # hunting_vision only

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


# ── Privileged teacher for hunting_vision (lead pursuit of a fleeing rabbit) ─
# info["privileged"] layout = the 12-dim target stream:
#   [0:3] dx,dy,dz  [3:6] vx,vy,vz (per step)  [6] dist/DIST_SCALE  [7] heading_err
#   [8] los_hit  [9] in_range  [10] life  [11] visible
def _wrap_deg(a):
    while a < -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def scripted_action_vision(vec, priv, cfg, mem):
    """Lead-pursuit hunter on privileged state. `mem` is a per-episode dict."""
    if priv is None or priv[11] < 0.5:
        return ACT_TURN_R                                   # spin until the prey is sensed
    dx, dy, dz = priv[0], priv[1], priv[2]
    vx, vz = priv[3], priv[5]
    dist = priv[6] * cfg.DIST_SCALE
    yaw, pitch = vec[1] * 180.0, vec[2] * 90.0

    if priv[9] > 0.5 and abs(priv[7]) < 0.12 and (priv[8] > 0.5 or cfg.AUTO_ATTACK):
        return ACT_ATTACK

    # Aim where the rabbit WILL be: lead by roughly the steps needed to get there.
    lead = min(dist / 1.0, 8.0)
    ax, az = dx + vx * lead, dz + vz * lead
    herr = _wrap_deg(-180.0 * math.atan2(ax, az) / math.pi - yaw)
    if abs(herr) > 12.0:
        return ACT_TURN_R if herr > 0 else ACT_TURN_L

    # Up close, pitch the crosshair onto a small, low target.
    if dist < 6.0:
        horiz = max(math.hypot(dx, dz), 0.3)
        mid = 0.5 * getattr(cfg, "TARGET_HEIGHT", 0.5)        # aim at the body's middle
        want_pitch = -180.0 * math.atan2((dy + mid) - 1.62, horiz) / math.pi
        if want_pitch - pitch > 10.0:
            return ACT_LOOK_D
        if pitch - want_pitch > 10.0:
            return ACT_LOOK_U

    # Natural terrain: jump when walking forward but not moving (1-block step).
    speed = math.hypot(vec[4], vec[5])
    mem["stuck"] = mem.get("stuck", 0) + 1 if (mem.get("moved") and speed < 0.05) else 0
    mem["moved"] = True
    if mem["stuck"] >= 2 and cfg.N_ACTIONS > ACT_JUMP_FWD:
        mem["stuck"] = 0
        return ACT_JUMP_FWD
    return ACT_SPRINT if dist > 4.0 else ACT_FORWARD


def run_vision(args, cfg):
    """Record frames + proprio demos with the privileged teacher."""
    from utils.demo_io import DemoWriter
    max_steps = args.max_steps or cfg.MAX_STEPS
    level = int(os.environ.get("MALMO_HUNT_LEVEL", cfg.HUNT_LEVEL))
    out_dir = args.output or os.path.join(
        os.path.dirname(PARKOUR_ROOT), "demos", "{0}_{1}_L{2}".format(args.env, cfg.PREY, level))
    writer = DemoWriter(out_dir, args.env)
    env = EnvClient(cfg.INPUT_SIZE, port=args.port, video=True,
                    frame_shape=(cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS))
    print("=" * 60)
    print("Scripted vision demo generator (privileged teacher)")
    print("  env {0}  prey {1}  level L{2} (must match the env server's MALMO_HUNT_LEVEL / MALMO_HUNT_PREY)".format(
        args.env, cfg.PREY, level))
    print("  output {0}  ({1} episodes already there)".format(out_dir, len(writer)))
    print("=" * 60)

    no_target = [0.0] * 6 + [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]   # the env's "not visible" stream
    kills = 0
    try:
        for ep_idx in range(args.episodes):
            vec, frame = env.reset()
            priv, mem, total, outcome = None, {}, 0.0, "timeout"
            writer.begin_episode()
            for _ in range(max_steps):
                action = scripted_action_vision(vec, priv, cfg, mem)
                (next_vec, next_frame), reward, done, info = env.step(action)
                writer.add(frame, vec, action, reward, done,
                           privileged=priv if priv is not None else no_target)
                total += reward
                vec, frame, priv = next_vec, next_frame, info.get("privileged")
                if done:
                    outcome = info.get("outcome", "unknown")
                    break
            name = writer.end_episode(outcome, source="scripted")
            kills += int(outcome == "killed")
            print("Episode {0}/{1}: {2}  reward {3:.1f}  -> {4}".format(
                ep_idx + 1, args.episodes, outcome, total, name))
    except KeyboardInterrupt:
        print("\nInterrupted (finished episodes are saved).")
    finally:
        env.close()
    print("\nKills this run: {0}/{1}. Demo set now has {2} episodes in {3}".format(
        kills, args.episodes, len(writer), out_dir))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate scripted hunting demos")
    parser.add_argument("--env", type=str, default="hunting", choices=list(ENV_CONFIGS.keys()),
                        help="Env the server runs (default: hunting). hunting_vision records "
                             "a frames demo directory with the privileged teacher.")
    parser.add_argument("--port",     type=int, default=9999,
                        help="Env server TCP port (default: 9999)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes to record (default: 30)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Output JSON path (default: demos/hunting.json); a directory "
                             "for hunting_vision (default: demos/hunting_vision_L<level>/)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (default: cfg.MAX_STEPS)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = ENV_CONFIGS[args.env]
    if getattr(cfg, "LEVELS", None):
        from training.configs.hunting_vision_cfg import apply_hunt_level
        apply_hunt_level(cfg)   # teacher reads AUTO_ATTACK / MAX_STEPS for the rung
    if getattr(cfg, "OBS_MODE", "full") == "proprio":
        return run_vision(args, cfg)
    cfg  = cfg()
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
        data = {"env": args.env, "episodes": []}

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
