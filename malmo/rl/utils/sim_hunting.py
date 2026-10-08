"""
utils/sim_hunting.py
--------------------
OFFLINE kinematic stand-in for the hunting env. NOT Minecraft.

Reproduces the 98-dim observation layout, the 11-action space, auto-attack and
the reward terms of envs/hunting_env.py closely enough to exercise the whole
world-model stack (train → imagine → compare) without a Malmo client. The
dynamics (speeds, turn rate, pig behaviour) are approximations, so anything
measured here validates PLUMBING, not model quality on the real game.

Interface matches HuntingEnv / EnvClient: reset() -> obs, step(a) -> (obs, r, done, info).
"""

import math
import numpy as np

# Approximate per-step (0.15 s) kinematics
WALK, SPRINT, STRAFE = 0.65, 0.84, 0.55
TURN_DEG, PITCH_DEG = 13.5, 6.75
PIG_SPEED = {"penned": 0.0, "wandering": 0.12, "fleeing": 0.2}
PIG_TURN_NOISE = {"penned": 0.0, "wandering": 0.5, "fleeing": 0.9}   # rad/step heading jitter


class SimHunting:
    def __init__(self, cfg, mode=None, seed=0):
        self.cfg = cfg
        self.mode = mode or cfg.TARGET_MODE
        self.rng = np.random.RandomState(seed)

    # ── gym-ish API ──────────────────────────────────────────────────────────
    def reset(self):
        c = self.cfg
        self.pos = np.array(c.SPAWN, dtype=np.float32)
        self.yaw = float(self.rng.uniform(-180, 180)) if self.mode != "penned" else 0.0
        self.pitch = 0.0
        self.prev_pos = self.pos.copy()
        if self.mode == "penned":
            tgt = self.pos + np.array([0, 0, c.PENNED_DZ], dtype=np.float32)
        elif self.mode == "wandering":
            tgt = self._rand_spawn(0.0)
        else:
            tgt = self._rand_spawn(c.FLEEING_MIN_DIST)
        self.tgt = tgt.astype(np.float32)
        self.prev_tgt = self.tgt.copy()
        self.tgt_heading = float(self.rng.uniform(-math.pi, math.pi))
        self.life = c.TARGET_MAX_LIFE
        self.steps = 0
        self.prev_phi = 0.0
        self.prev_dist = self._dist()
        self.on_ground = 1.0
        return self._obs()

    def step(self, a):
        c = self.cfg
        a = int(a)
        name = c.ACTIONS[a][0]
        yr = math.radians(self.yaw)
        fwd = np.array([-math.sin(yr), 0.0, math.cos(yr)], dtype=np.float32)
        right = np.array([-math.cos(yr), 0.0, -math.sin(yr)], dtype=np.float32)
        self.prev_pos = self.pos.copy()
        self.prev_tgt = self.tgt.copy()
        move = np.zeros(3, dtype=np.float32)
        if name == "move_forward":    move = fwd * WALK
        elif name == "move_backward": move = -fwd * WALK
        elif name == "strafe_left":   move = -right * STRAFE
        elif name == "strafe_right":  move = right * STRAFE
        elif name == "sprint_forward": move = fwd * SPRINT
        elif name == "turn_left":     self.yaw -= TURN_DEG
        elif name == "turn_right":    self.yaw += TURN_DEG
        elif name == "look_up":       self.pitch = max(-90.0, self.pitch - PITCH_DEG)
        elif name == "look_down":     self.pitch = min(90.0, self.pitch + PITCH_DEG)
        self.yaw = (self.yaw + 180.0) % 360.0 - 180.0
        self.pos = self.pos + move
        self.pos[0] = np.clip(self.pos[0], c.ARENA_MIN, c.ARENA_MAX)
        self.pos[2] = np.clip(self.pos[2], c.ARENA_MIN, c.ARENA_MAX)

        self._move_pig()

        # Attack: explicit, or auto-fire when aligned and in range
        herr = self._heading_error()
        in_range = self._dist() <= c.ATTACK_RANGE
        attacking = name == "attack" or (
            getattr(c, "AUTO_ATTACK", False) and in_range and abs(herr) < c.AUTO_ATTACK_ANGLE)
        prev_life = self.life
        if attacking and in_range and abs(herr) < 0.25 and self.rng.rand() < 0.5:
            self.life = max(0.0, self.life - 1.0)
            away = self.tgt - self.pos
            away[1] = 0
            n = np.linalg.norm(away) + 1e-6
            self.tgt = self.tgt + away / n * 0.4          # knockback

        self.steps += 1
        reward, done, outcome = self._reward(prev_life)
        self.prev_dist = self._dist()
        info = {"outcome": outcome, "steps": self.steps, "pos": self.pos.copy(), "action": a}
        return self._obs(), reward, done, info

    # ── internals ────────────────────────────────────────────────────────────
    def _rand_spawn(self, min_dist):
        c = self.cfg
        for _ in range(100):
            p = self.rng.uniform(c.ARENA_MIN, c.ARENA_MAX, size=2)
            if np.hypot(p[0] - self.pos[0], p[1] - self.pos[2]) >= max(min_dist, 2.0):
                return np.array([p[0], self.pos[1], p[1]])
        return np.array([c.ARENA_MAX, self.pos[1], c.ARENA_MAX])

    def _move_pig(self):
        c = self.cfg
        sp = PIG_SPEED[self.mode]
        if sp == 0.0:
            return
        if self.mode == "fleeing" and self._dist() < 6.0:
            away = self.tgt - self.pos
            self.tgt_heading = math.atan2(away[2], away[0]) + self.rng.normal(0, 0.4)
        else:
            self.tgt_heading += self.rng.normal(0, PIG_TURN_NOISE[self.mode])
        step = sp * np.array([math.cos(self.tgt_heading), 0, math.sin(self.tgt_heading)],
                             dtype=np.float32)
        self.tgt = self.tgt + step
        for ax in (0, 2):
            if self.tgt[ax] < c.ARENA_MIN or self.tgt[ax] > c.ARENA_MAX:
                self.tgt[ax] = np.clip(self.tgt[ax], c.ARENA_MIN, c.ARENA_MAX)
                self.tgt_heading = math.pi - self.tgt_heading if ax == 0 else -self.tgt_heading

    def _delta(self):
        return self.tgt - self.pos

    def _dist(self):
        return float(np.linalg.norm(self._delta()))

    def _heading_error(self):
        dx, _, dz = self._delta()
        desired = -180.0 * math.atan2(dx, dz) / math.pi
        h = desired - self.yaw
        while h < -180.0: h += 360.0
        while h > 180.0: h -= 360.0
        return h / 180.0

    def _reward(self, prev_life):
        c = self.cfg
        if self.life <= 0.0:
            return c.REWARD_KILL, True, "killed"
        r = c.REWARD_STEP_PENALTY
        if self.life < prev_life:
            r += c.REWARD_HIT * (prev_life - self.life)
        delta = float(np.clip(self.prev_dist - self._dist(), -c.APPROACH_CLIP, c.APPROACH_CLIP))
        r += c.REWARD_APPROACH_COEF * delta
        herr = self._heading_error()
        los = abs(herr) < 0.04 and self._dist() <= 20
        if los:
            r += c.REWARD_AIM
        if getattr(c, "AIM_ALIGN_COEF", 0.0):
            phi = math.cos(herr * math.pi)
            r += c.AIM_ALIGN_COEF * (c.GAMMA * phi - self.prev_phi)
            self.prev_phi = phi
        if self.steps >= c.MAX_STEPS:
            return r + c.REWARD_TIMEOUT, True, "timeout"
        return r, False, "alive"

    def _voxels(self):
        c = self.cfg
        bx, bz = int(math.floor(self.pos[0])), int(math.floor(self.pos[2]))
        g = np.zeros((c.GRID_X, c.GRID_Y, c.GRID_Z), dtype=np.float32)
        lo, hi = c.ARENA_MIN - 1, c.ARENA_MAX
        for i in range(c.GRID_X):
            for k in range(c.GRID_Z):
                x, z = bx + i - 2, bz + k - 2
                inside = lo <= x <= hi and lo <= z <= hi
                g[i, 0, k] = 1.0 if inside else 0.0          # floor (y-1)
                if x in (lo, hi) or z in (lo, hi):           # boundary wall
                    g[i, 1, k] = g[i, 2, k] = 1.0
        return g.reshape(-1)

    def _obs(self):
        c = self.cfg
        vel = self.pos - self.prev_pos
        tvel = self.tgt - self.prev_tgt
        d = self._delta()
        dist = self._dist()
        herr = self._heading_error()
        los = 1.0 if (abs(herr) < 0.04 and dist <= 20) else 0.0
        proprio = np.array([
            self.on_ground, self.yaw / 180.0, self.pitch / 90.0,
            vel[1], vel[0], vel[2],
            *(self.pos - np.array(c.SPAWN, dtype=np.float32)),
            1.0, 1.0,
        ], dtype=np.float32)
        target = np.array([
            d[0], d[1], d[2], tvel[0], tvel[1], tvel[2],
            dist / c.DIST_SCALE, herr, los, 1.0 if dist <= c.ATTACK_RANGE else 0.0,
            self.life / c.TARGET_MAX_LIFE, 1.0,
        ], dtype=np.float32)
        return np.concatenate([proprio, target, self._voxels()]).astype(np.float32)


def scripted_hunter(obs, cfg, rng, eps=0.0):
    """Heuristic hunter (same logic as utils/generate_hunting_demos.py)."""
    if rng.rand() < eps:
        return int(rng.randint(cfg.N_ACTIONS))
    p = cfg.PROPRIOCEPTION_SIZE
    herr, in_range = obs[p + 7], obs[p + 9]
    if abs(herr) > 0.12:
        return 6 if herr > 0 else 5            # turn_right / turn_left
    return 9 if in_range else 4                # attack / sprint_forward
