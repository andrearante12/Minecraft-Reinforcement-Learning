"""
_gen_sample_logs.py
-------------------
Generates sample log files for testing visualization/replay.py.

Produces three files with the timestamp 20260401_143022:
  simple_jump_ppo_20260401_143022_trajectories.csv
  simple_jump_ppo_20260401_143022_episodes.csv
  simple_jump_ppo_20260401_143022_updates.csv

Run from the logs/ directory:
    python _gen_sample_logs.py
"""

import csv
import math
import os
import random

random.seed(42)

TIMESTAMP = "20260401_143022"
PREFIX    = f"sample_simple_jump_ppo_{TIMESTAMP}"
DIR       = os.path.dirname(os.path.abspath(__file__))

ACTIONS = [
    "forward", "backward", "left", "right",
    "sprint_forward", "jump", "sprint_jump", "jump_forward",
    "sprint_jump_left", "sprint_jump_right",
    "look_down", "look_up", "turn_left", "turn_right", "no_op",
]

# ── Blocks: spawn platform z=2..4, gap z=5, goal platform z=6..8 ─────────────
SPAWN_PLATFORM_BLOCKS = [
    (x, 45, z, "stone")
    for x in range(-1, 3)
    for z in range(2, 5)
]
GOAL_PLATFORM_BLOCKS = [
    (x, 45, z, "stone")
    for x in range(-1, 3)
    for z in range(6, 9)
]
ALL_BLOCKS = SPAWN_PLATFORM_BLOCKS + GOAL_PLATFORM_BLOCKS

SPAWN = (0.5, 46.0, 3.5)
GOAL  = (0.5, 45.0, 6.5)

STEP_PENALTY  = -0.01
FELL_REWARD   = -5.0
TIMEOUT_REWARD = -5.0
SUCCESS_REWARD = 10.0
LANDING_TICK_REWARD = 0.5
FALL_Y = 43.0
MAX_STEPS = 30
LANDING_TICKS = 5   # steps on goal block before success


# ── Trajectory generators ─────────────────────────────────────────────────────

def _make_timeout_episode(ep_num, noise=0.0):
    """Agent wanders without making meaningful progress — exhausts step budget."""
    steps = []
    x, y, z = SPAWN
    yaw = 180.0  # facing +z
    pitch = 0.0
    on_ground = True

    action_pool = ["no_op", "turn_left", "turn_right", "forward", "backward", "no_op", "look_up"]
    total_reward = 0.0

    for s in range(MAX_STEPS):
        act = random.choice(action_pool)
        reward = STEP_PENALTY

        if act == "forward":
            z += 0.18 + random.uniform(-0.02, 0.02)
        elif act == "backward":
            z -= 0.12
        elif act == "turn_left":
            yaw = (yaw - 15 + 360) % 360
        elif act == "turn_right":
            yaw = (yaw + 15) % 360
        elif act == "look_up":
            pitch = max(pitch - 5, -45)

        # Keep on spawn platform
        z = max(2.2, min(z, 4.8))
        x = max(-0.5, min(x + random.uniform(-0.05, 0.05), 1.5))

        done = (s == MAX_STEPS - 1)
        outcome = "timeout" if done else "alive"
        if done:
            reward += TIMEOUT_REWARD
        total_reward += reward

        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "yaw": round(yaw, 1), "pitch": round(pitch, 1),
            "on_ground": int(on_ground), "action": act,
            "reward": round(reward, 4), "done": int(done),
            "outcome": outcome, "env": "simple_jump",
        })

    return steps, round(total_reward, 4), MAX_STEPS, "timeout"


def _make_fell_episode(ep_num):
    """Agent sprints toward the gap but never jumps — falls in."""
    steps = []
    x, y, z = SPAWN
    yaw = 180.0
    pitch = 0.0
    on_ground = True
    total_reward = 0.0

    # Phase 1: sprint toward gap
    phase1_steps = random.randint(6, 10)
    for s in range(phase1_steps):
        act = "sprint_forward"
        z += 0.38 + random.uniform(-0.03, 0.03)
        x += random.uniform(-0.02, 0.02)
        reward = STEP_PENALTY
        total_reward += reward
        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "yaw": yaw, "pitch": pitch, "on_ground": 1, "action": act,
            "reward": round(reward, 4), "done": 0, "outcome": "alive", "env": "simple_jump",
        })
        if z >= 5.0:
            break

    # Phase 2: fall (no block under agent at z=5)
    fall_steps = random.randint(3, 6)
    on_ground = False
    for i in range(fall_steps):
        s = len(steps)
        act = "no_op"
        z += 0.1
        y -= 0.8 + i * 0.4   # gravity accelerates fall
        reward = STEP_PENALTY
        done = (y <= FALL_Y) or (i == fall_steps - 1 and y <= FALL_Y + 1.5)
        if done:
            y = min(y, FALL_Y - 0.1)
            reward += FELL_REWARD
        outcome = "fell" if done else "alive"
        total_reward += reward
        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "yaw": yaw, "pitch": pitch, "on_ground": 0, "action": act,
            "reward": round(reward, 4), "done": int(done), "outcome": outcome,
            "env": "simple_jump",
        })
        if done:
            break

    n_steps = len(steps)
    return steps, round(total_reward, 4), n_steps, "fell"


def _make_landed_episode(ep_num):
    """Agent sprints then sprint-jumps across the gap."""
    steps = []
    x, y, z = SPAWN
    yaw = 180.0
    pitch = 0.0
    on_ground = True
    total_reward = 0.0
    s = 0

    # Phase 1: a couple of sprint steps
    sprint_steps = random.randint(2, 4)
    for _ in range(sprint_steps):
        act = "sprint_forward"
        z += 0.35 + random.uniform(-0.02, 0.02)
        reward = STEP_PENALTY
        total_reward += reward
        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "yaw": yaw, "pitch": pitch, "on_ground": 1, "action": act,
            "reward": round(reward, 4), "done": 0, "outcome": "alive", "env": "simple_jump",
        })
        s += 1

    # Phase 2: sprint jump — arc over the gap
    arc_steps = 4
    arc_x = [0.0, 0.35, 0.65, 0.95]  # forward progress each step
    arc_y = [0.4, 0.2, -0.1, -0.4]   # height delta
    for i in range(arc_steps):
        act = "sprint_jump" if i == 0 else "no_op"
        z += arc_x[i] + random.uniform(-0.02, 0.02)
        y += arc_y[i]
        on_ground_step = (i == arc_steps - 1)
        if on_ground_step:
            y = 46.0   # land on goal platform surface
        reward = STEP_PENALTY
        total_reward += reward
        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x + random.uniform(-0.05, 0.05), 3),
            "y": round(y, 3), "z": round(z, 3),
            "yaw": yaw, "pitch": pitch,
            "on_ground": int(on_ground_step), "action": act,
            "reward": round(reward, 4), "done": 0, "outcome": "alive", "env": "simple_jump",
        })
        s += 1

    # Phase 3: landing ticks on goal block
    for tick in range(LANDING_TICKS):
        act = "no_op"
        reward = STEP_PENALTY + LANDING_TICK_REWARD   # +0.5 per landing tick
        done = (tick == LANDING_TICKS - 1)
        if done:
            reward += SUCCESS_REWARD
        outcome = "landed" if done else "alive"
        total_reward += reward
        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x + random.uniform(-0.02, 0.02), 3),
            "y": 46.0, "z": round(z + random.uniform(-0.02, 0.02), 3),
            "yaw": yaw, "pitch": pitch, "on_ground": 1, "action": act,
            "reward": round(reward, 4), "done": int(done), "outcome": outcome,
            "env": "simple_jump",
        })
        s += 1

    n_steps = s
    return steps, round(total_reward, 4), n_steps, "landed"


def _make_near_miss_episode(ep_num):
    """Agent jumps but lands just short of the goal and times out near the edge."""
    steps = []
    x, y, z = SPAWN
    yaw = 180.0
    pitch = 0.0
    total_reward = 0.0
    s = 0

    # Sprint + jump but land at z~5.7 (short of Z_SUCCESS=6.0)
    for _ in range(3):
        act = "sprint_forward"
        z += 0.33
        steps.append({
            "episode": ep_num, "step": s, "x": round(x, 3), "y": round(y, 3),
            "z": round(z, 3), "yaw": yaw, "pitch": pitch, "on_ground": 1,
            "action": act, "reward": STEP_PENALTY, "done": 0, "outcome": "alive",
            "env": "simple_jump",
        })
        total_reward += STEP_PENALTY
        s += 1

    # Under-powered jump — lands at z=5.7
    arc_x = [0.28, 0.22, 0.12, 0.05]
    arc_y_d = [0.4, 0.1, -0.3, -0.5]
    for i in range(4):
        act = "sprint_jump" if i == 0 else "no_op"
        z += arc_x[i]
        y += arc_y_d[i]
        on_ground_step = (i == 3)
        if on_ground_step:
            y = 46.0
        steps.append({
            "episode": ep_num, "step": s,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "yaw": yaw, "pitch": pitch, "on_ground": int(on_ground_step),
            "action": act, "reward": STEP_PENALTY, "done": 0, "outcome": "alive",
            "env": "simple_jump",
        })
        total_reward += STEP_PENALTY
        s += 1

    # Rest of steps: stuck near edge, timeout
    remaining = MAX_STEPS - s
    for i in range(remaining):
        act = random.choice(["forward", "no_op", "jump"])
        done = (i == remaining - 1)
        reward = STEP_PENALTY + (TIMEOUT_REWARD if done else 0)
        outcome = "timeout" if done else "alive"
        total_reward += reward
        steps.append({
            "episode": ep_num, "step": s, "x": round(x, 3), "y": 46.0,
            "z": round(z + random.uniform(-0.05, 0.05), 3),
            "yaw": yaw, "pitch": pitch, "on_ground": 1, "action": act,
            "reward": round(reward, 4), "done": int(done), "outcome": outcome,
            "env": "simple_jump",
        })
        total_reward += 0  # already added above
        s += 1

    # Fix double-counting
    total_reward = sum(st["reward"] for st in steps)
    return steps, round(total_reward, 4), s, "timeout"


# ── Episode schedule (20 episodes showing learning curve) ────────────────────

def build_episodes():
    """
    Learning progression over 20 episodes:
      1-4   : timeout (agent hasn't found useful policy)
      5-7   : fell   (exploring gap direction)
      8-9   : near-miss / timeout
      10-12 : fell then landed (first successes)
      13-20 : mostly landed
    """
    schedule = [
        ("timeout", 1), ("timeout", 2), ("timeout", 3), ("timeout", 4),
        ("fell",    5), ("fell",    6), ("fell",    7),
        ("near",    8), ("timeout", 9),
        ("fell",   10), ("landed", 11), ("landed", 12),
        ("fell",   13), ("landed", 14), ("landed", 15),
        ("landed", 16), ("landed", 17), ("landed", 18),
        ("landed", 19), ("landed", 20),
    ]

    all_steps   = []
    ep_summaries = []

    for kind, ep_num in schedule:
        if kind == "timeout":
            steps, total_rew, n, outcome = _make_timeout_episode(ep_num)
        elif kind == "fell":
            steps, total_rew, n, outcome = _make_fell_episode(ep_num)
        elif kind == "landed":
            steps, total_rew, n, outcome = _make_landed_episode(ep_num)
        else:
            steps, total_rew, n, outcome = _make_near_miss_episode(ep_num)

        all_steps.extend(steps)
        ep_summaries.append((ep_num, total_rew, n, outcome))

    return all_steps, ep_summaries


# ── PPO update schedule ───────────────────────────────────────────────────────

def build_updates(n_episodes=20):
    """
    One PPO update every ~2 episodes (N_STEPS filled after ~2 * ~20 env steps).
    ~10 updates total, losses decreasing as agent improves.
    """
    rows = []
    for i in range(1, 11):
        t = (i - 1) / 9.0   # 0 → 1 as training progresses
        policy_loss = round(0.42 - 0.35 * t + random.uniform(-0.02, 0.02), 6)
        value_loss  = round(0.68 - 0.51 * t + random.uniform(-0.03, 0.03), 6)
        entropy     = round(2.58 - 0.85 * t + random.uniform(-0.05, 0.05), 6)
        kl_div      = round(0.012 + random.uniform(-0.002, 0.004), 6)
        rows.append({
            "update": i,
            "policy_loss": policy_loss,
            "value_loss":  value_loss,
            "entropy":     entropy,
            "kl_div":      kl_div,
        })
    return rows


# ── Writers ───────────────────────────────────────────────────────────────────

def write_trajectories(path, all_steps):
    with open(path, "w", newline="") as fh:
        fh.write(f"# env: simple_jump\n")
        for bx, by, bz, bt in ALL_BLOCKS:
            fh.write(f"# block: {bx},{by},{bz},{bt}\n")
        fh.write(f"# spawn: {SPAWN[0]},{SPAWN[1]},{SPAWN[2]}\n")
        fh.write(f"# goal: {GOAL[0]},{GOAL[1]},{GOAL[2]}\n")

        writer = csv.DictWriter(fh, fieldnames=[
            "episode", "step", "x", "y", "z",
            "yaw", "pitch", "on_ground", "action",
            "reward", "done", "outcome", "env",
        ])
        writer.writeheader()
        writer.writerows(all_steps)
    print(f"  Wrote {len(all_steps):>4} rows -> {os.path.basename(path)}")


def write_episodes(path, summaries):
    import time as _time
    base_ts = 51802   # seconds since midnight = ~14:23:22
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "reward", "steps", "outcome", "env", "timestamp"])
        for ep, reward, steps, outcome in summaries:
            secs = base_ts + ep * 8
            h    = secs // 3600
            m    = (secs % 3600) // 60
            s    = secs % 60
            ts   = f"{h:02d}:{m:02d}:{s:02d}"
            writer.writerow([ep, reward, steps, outcome, "simple_jump", ts])
    print(f"  Wrote {len(summaries):>4} rows -> {os.path.basename(path)}")


def write_updates(path, updates):
    with open(path, "w", newline="") as fh:
        if not updates:
            return
        writer = csv.DictWriter(fh, fieldnames=list(updates[0].keys()))
        writer.writeheader()
        writer.writerows(updates)
    print(f"  Wrote {len(updates):>4} rows -> {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating sample logs in: {DIR}")
    print(f"  Run prefix: {PREFIX}\n")

    all_steps, ep_summaries = build_episodes()
    updates                 = build_updates()

    write_trajectories(os.path.join(DIR, f"{PREFIX}_trajectories.csv"), all_steps)
    write_episodes(    os.path.join(DIR, f"{PREFIX}_episodes.csv"),      ep_summaries)
    write_updates(     os.path.join(DIR, f"{PREFIX}_updates.csv"),       updates)

    print(f"\nDone. Load with:")
    print(f"  python malmo/rl/visualization/replay.py --run malmo/rl/logs/{PREFIX}")
