"""
envs/hunting_env.py
-------------------
Gym-style Malmo wrapper for the hunting task. The agent must locate, aim at,
approach, and attack a passive animal (default Pig) until it dies.

This is the headline showcase environment for the world-model agent: a
long-horizon, sparse-reward task whose dynamics include a MOVING TARGET (an
independent animal). The target's predictability is controlled by
cfg.TARGET_MODE (penned / wandering / fleeing) — the central research knob.

Mirrors BridgingEnv's structure and Malmo plumbing. Key differences:
  - Target animal spawned per-episode via DrawEntity (position from TARGET_MODE)
  - Observation includes the nearest animal's relative position AND velocity,
    distance, heading error, line-of-sight hit, attack range, and life
  - Reward derived from the target's life drop (hits) + approach + a dominant
    terminal kill bonus
  - Success = target killed (life reaches 0 / disappears after being damaged)

Observation vector layout (INPUT_SIZE = 98):
    [0]      onGround
    [1]      yaw              (normalized -1..1)
    [2]      pitch            (normalized -1..1)
    [3..5]   delta_y/x/z      (velocity)
    [6..8]   pos_x/y/z        (relative to spawn)
    [9]      food             (normalized 0..1)
    [10]     health           (normalized 0..1)
    [11..13] tgt_dx/dy/dz     (target - agent)
    [14..16] tgt_vx/vy/vz     (target velocity, finite diff)
    [17]     tgt_dist         (normalized)
    [18]     heading_error    (normalized -1..1; 0 = facing the target)
    [19]     los_hit          (1 if crosshair is on the animal)
    [20]     in_range         (1 if within attack range)
    [21]     tgt_life         (normalized 0..1)
    [22]     tgt_visible      (1 if target within sensor range)
    [23:98]  voxel grid       (5 x 3 x 5 = 75 values)
"""

import sys
import os
import math
import time
import json
import random
import re
import numpy as np

# ── Add rl/ root to path so training.configs is importable ────────────────────
PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from training.configs.base_cfg import BaseCFG

# ── Malmo import ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(BaseCFG.MALMO_PYTHON))

try:
    import MalmoPython
except ImportError:
    raise ImportError(
        "Could not import MalmoPython.\n"
        "Tried: {0}".format(os.path.abspath(BaseCFG.MALMO_PYTHON))
    )


class HuntingEnv:
    def __init__(self, cfg, malmo_port=None, force_reset=False):
        # Curriculum rung (hunting_vision): mutates cfg before anything reads it.
        if getattr(cfg, "LEVELS", None):
            from training.configs.hunting_vision_cfg import apply_hunt_level
            print("  [hunting] level L{0}".format(apply_hunt_level(cfg)))
        self.cfg       = cfg
        self.actions   = cfg.ACTIONS
        self.n_actions = cfg.N_ACTIONS
        self._malmo_port = malmo_port if malmo_port is not None else cfg.MALMO_PORT

        self._prev_pos = np.array(cfg.SPAWN, dtype=np.float32)
        self._steps    = 0

        # Target-tracking state (updated each step in _build_obs_vector)
        self._prev_tgt_pos  = None
        self._tgt_pos       = None
        self._tgt_dist      = float(cfg.DIST_SCALE)
        self._tgt_life      = cfg.TARGET_MAX_LIFE
        self._tgt_visible   = False
        self._los_hit       = False
        self._target_seen   = False     # ever seen the animal this episode
        self._damaged       = False     # ever dealt damage this episode
        self._heading_error = 0.0       # normalized heading error to target
        self._in_range      = False     # within ATTACK_RANGE
        self._prev_phi      = 0.0      # previous potential Φ = cos(heading_error·π)
        self._prev_phi_p    = 0.0      # previous pitch potential
        self._pit_steps     = 0        # consecutive steps spent in a pit
        self._pitch_error   = 0.0      # (desired - actual pitch) / 90, aiming at body middle
        self._pitch_deg     = 0.0
        self._n_targets     = 0        # target-mob count last tick (multi-target kills)
        self._last_target   = None     # 12-dim target stream (privileged in proprio mode)
        self._dmg_signal    = 0.0      # Malmo RewardForDamagingEntity total since last step
        self._kills         = 0        # kills this episode (KILLS_TO_END > 1)
        self._reacquire     = False    # just killed one: skip approach shaping once

        # Video observation (off by default; see training/configs/hunting_video_cfg.py).
        # MALMO_RECORD_VIDEO overrides size for demo-site clips.
        from envs.video_record import apply_record_override
        self._video     = apply_record_override(cfg)
        self.last_frame = None

        self.observation_shape = (cfg.INPUT_SIZE,)

        # Seeded natural worlds: a scouted heightmap (utils/scout_seeds.py) puts
        # the agent and prey ON the ground instead of at a fixed y.
        self._hm = None
        if getattr(cfg, "HEIGHTMAP", None):
            self._load_heightmap(cfg.HEIGHTMAP)

        with open(cfg.MISSION_FILE, "r") as f:
            self._mission_xml_base = f.read()

        self._agent_host = MalmoPython.AgentHost()
        try:
            self._agent_host.parse(["env_server"])
        except RuntimeError as e:
            print("ERROR parsing AgentHost:", e)
            sys.exit(1)

    # ── Public API ───────────────────────────────────────────────────────────
    def reset(self):
        self._steps        = 0
        self._prev_pos      = np.array(self.cfg.SPAWN, dtype=np.float32)
        self._prev_tgt_pos  = None
        self._tgt_pos       = None
        self._tgt_dist      = float(self.cfg.DIST_SCALE)
        self._tgt_life      = self.cfg.TARGET_MAX_LIFE
        self._tgt_visible   = False
        self._los_hit       = False
        self._target_seen   = False
        self._damaged       = False
        self._heading_error = 0.0
        self._in_range      = False
        self._prev_phi      = 0.0
        self._prev_phi_p    = 0.0
        self._pit_steps     = 0
        self._pitch_error   = 0.0
        self._pitch_deg     = 0.0
        self._n_targets     = 0
        self._last_target   = None
        self._dmg_signal    = 0.0
        self._kills         = 0
        self._reacquire     = False
        self.last_frame     = None   # never leak a frame across an episode boundary

        self._pick_world()
        spawns = [self._choose_target_spawn()
                  for _ in range(int(getattr(self.cfg, "N_TARGETS", 1)))]
        mission_xml = self._build_mission_xml(spawns)

        time.sleep(0.5)
        self._start_mission(mission_xml)
        print("  [reset] target {mob} at {pos}  mode={mode}".format(
            mob=self.cfg.TARGET_MOB, mode=self.cfg.TARGET_MODE,
            pos=" ".join("({0:.1f}, {1:.1f})".format(x, z) for x, z in spawns)))
        return self._get_observation()

    def _build_mission_xml(self, spawns):
        """Mission XML for one episode: cage (if any) + one DrawEntity per spawn."""
        ty = float(getattr(self.cfg, "TARGET_SPAWN_Y", 46.0))
        draw = self._cage_xml() + self._fill_water_xml() + getattr(self, "_hazard_xml", "")
        for tx, tz in spawns:
            y = self._feet_y(tx, tz) if self._hm is not None else ty
            draw += '    <DrawEntity x="{x:.1f}" y="{y:.1f}" z="{z:.1f}" type="{mob}"/>\n'.format(
                x=tx, y=y, z=tz, mob=self.cfg.TARGET_MOB)
        mission_xml = self._mission_xml_base.replace(
            '</DrawingDecorator>', draw + '      </DrawingDecorator>')
        if getattr(self.cfg, "USE_DAMAGE_SIGNAL", False):
            # The damage event must be for the configured prey.
            mission_xml = re.sub(r'<Mob type="[^"]*"', '<Mob type="{0}"'.format(self.cfg.TARGET_MOB),
                                 mission_xml)
        seed = getattr(self.cfg, "WORLD_SEED", None)
        if seed is not None:
            mission_xml = re.sub(r'seed="[^"]*"', 'seed="{0}"'.format(seed), mission_xml)
        if self._hm is not None:
            sx, sy, sz = self.cfg.SPAWN
            mission_xml = re.sub(
                r'<Placement x="[^"]*" y="[^"]*" z="[^"]*"',
                '<Placement x="{0}" y="{1}" z="{2}"'.format(sx, sy, sz), mission_xml)
        return mission_xml

    # ── Heightmap (seeded natural worlds) ────────────────────────────────────
    _GROUND = ("grass", "dirt", "sand", "gravel", "stone", "clay", "mycelium",
               "sandstone", "hardened_clay", "stained_hardened_clay", "podzol")
    _PLANTS = ("tallgrass", "double_plant", "red_flower", "yellow_flower",
               "deadbush", "snow_layer", "brown_mushroom", "red_mushroom")

    def _load_heightmap(self, path):
        with open(path) as f:
            hm = json.load(f)
        self._hm = hm
        self._hm_top = np.array(hm["top_y"])
        half = hm["half"]
        # Standable columns: plain ground, or ground under grass/flowers.
        self._hm_ok = [(x, z) for z in range(-half, half + 1) for x in range(-half, half + 1)
                       if self._hm_block(x, z) in self._GROUND + self._PLANTS]
        # Agent spawn: nearest standable column to the origin.
        x0, z0 = min(self._hm_ok, key=lambda c: (c[0] ** 2 + c[1] ** 2, c))
        self.cfg.SPAWN = (x0 + 0.5, self._feet_y(x0 + 0.5, z0 + 0.5), z0 + 0.5)
        self.cfg.FALL_Y_THRESHOLD = float(self._hm_top.min()) - 4.0
        self._prev_pos = np.array(self.cfg.SPAWN, dtype=np.float32)
        print("  [hunting] heightmap {0}: spawn {1}, {2} standable columns".format(
            os.path.basename(path), self.cfg.SPAWN, len(self._hm_ok)))

    def _pick_world(self):
        """Per-episode world: random seed from WORLD_SEEDS + random trap holes.

        Reloads that seed's heightmap (spawn, standable columns), then carves
        HAZARDS dry pits / water-floored ponds the agent cannot climb out of.
        The heightmap is patched to match, so pit detection and target
        placement see the holes.
        """
        seeds = getattr(self.cfg, "WORLD_SEEDS", None)
        if seeds and self.cfg.HEIGHTMAP:
            seed = random.choice(seeds)
            self.cfg.WORLD_SEED = seed
            self._load_heightmap(os.path.join(os.path.dirname(self.cfg.HEIGHTMAP),
                                              "{0}.json".format(seed)))
        elif self._hm is not None:
            self._load_heightmap(self.cfg.HEIGHTMAP)    # fresh copy: holes are per-episode
        self._hazard_xml, self._hazard_cells = "", set()
        lo_n, hi_n = getattr(self.cfg, "HAZARDS", (0, 0))
        if self._hm is None or hi_n <= 0:
            return
        h, ok = self._hm["half"], set(self._hm_ok)
        self.cfg.FALL_Y_THRESHOLD = float(self._hm_top.min()) - 10.0
        sx, _, sz = self.cfg.SPAWN
        lo, hi = self.cfg.ARENA_MIN + 2, self.cfg.ARENA_MAX - 2
        placed = []
        for _ in range(random.randint(lo_n, hi_n)):
            for _try in range(40):
                cx, cz = random.randint(int(lo), int(hi)), random.randint(int(lo), int(hi))
                cells = [(cx + dx, cz + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
                if not all(c in ok for c in cells):
                    continue
                if max(abs(cx - sx), abs(cz - sz)) < 6:
                    continue
                if any(max(abs(cx - px), abs(cz - pz)) < 5 for px, pz in placed):
                    continue
                gs = [self._ground_y(x, z) for x, z in cells]
                if max(gs) - min(gs) > 1:
                    continue
                g, pond = min(gs), random.random() < 0.5
                top = self._hm["top_block"]
                self._hazard_xml += ('    <DrawCuboid x1="{0}" y1="{1}" z1="{2}" x2="{3}" y2="{4}" z2="{5}" '
                                     'type="air"/>\n').format(cx - 1, g - (4 if pond else 3), cz - 1,
                                                              cx + 1, g + 1, cz + 1)
                if pond:
                    self._hazard_xml += ('    <DrawCuboid x1="{0}" y1="{1}" z1="{2}" x2="{3}" y2="{4}" z2="{5}" '
                                         'type="water"/>\n').format(cx - 1, g - 4, cz - 1,
                                                                    cx + 1, g - 3, cz + 1)
                for x, z in cells:
                    self._hm_top[z + h][x + h] = g - 3 if pond else g - 4
                    top[z + h][x + h] = "water" if pond else "dirt"
                    self._hazard_cells.add((x, z))
                placed.append((cx, cz))
                break
        self._hm_ok = [c for c in self._hm_ok if c not in self._hazard_cells]
        if placed:
            print("  [world] seed {0}: {1} hazard(s) at {2}".format(
                self.cfg.WORLD_SEED, len(placed), placed))

    def _ground_y(self, x, z):
        """y of the solid ground block in column (x, z) (below any plant)."""
        top = int(self._hm_top[z + self._hm["half"]][x + self._hm["half"]])
        return top - 1 if self._hm_block(x, z) in self._PLANTS else top

    def _hm_block(self, x, z):
        h = self._hm["half"]
        return self._hm["top_block"][int(z) + h][int(x) + h]

    def _feet_y(self, x, z):
        """y where an entity standing on column (x, z) has its feet."""
        bx, bz = int(math.floor(x)), int(math.floor(z))
        top = int(self._hm_top[bz + self._hm["half"]][bx + self._hm["half"]])
        return float(top if self._hm_block(bx, bz) in self._PLANTS else top + 1)

    def _pit_depth(self, x, z, y):
        """Blocks the agent's feet are below the highest ground 2-3 columns away."""
        h = self._hm["half"]
        best = None
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                if max(abs(dx), abs(dz)) < 2:
                    continue
                bx, bz = int(math.floor(x)) + dx, int(math.floor(z)) + dz
                if abs(bx) > h or abs(bz) > h:
                    continue
                if self._hm_block(bx, bz) not in self._GROUND + self._PLANTS:
                    continue    # canopy, water, mushroom: not ground
                fy = self._feet_y(bx + 0.5, bz + 0.5)
                best = fy if best is None else max(best, fy)
        return 0.0 if best is None else best - y

    def _fill_water_xml(self):
        """Grass over every water column, flush with its lowest ground neighbour.

        The natural pond is a 2-block-deep trap right beside the spawn; the agent
        cannot climb out, so most episodes died there. (FILL_WATER; needs HEIGHTMAP.)
        """
        if not getattr(self.cfg, "FILL_WATER", False) or self._hm is None:
            return ""
        h, out = self._hm["half"], ""
        for z in range(-h, h + 1):
            for x in range(-h, h + 1):
                if "water" not in self._hm_block(x, z):
                    continue
                tops = [int(self._hm_top[z + dz + h][x + dx + h])
                        for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                        if abs(x + dx) <= h and abs(z + dz) <= h
                        and self._hm_block(x + dx, z + dz) in self._GROUND]
                lvl = min(tops) if tops else int(self._hm_top[z + h][x + h]) + 2
                out += ('    <DrawCuboid x1="{x}" y1="{y0}" z1="{z}" x2="{x}" y2="{y1}" z2="{z}" '
                        'type="grass"/>\n').format(x=x, z=z, y0=int(self._hm_top[z + h][x + h]) - 2, y1=lvl)
        return out

    def _cage_xml(self):
        """Boundary of half-width CAGE_HALF around the origin (None = none).

        CAGE_STYLE "bedrock" (default): bedrock walls + floor.
        CAGE_STYLE "fence": a see-through oak-fence pasture, one post per
        perimeter column placed on that column's ground (needs HEIGHTMAP).
        Neither cows nor the agent can jump a fence.
        """
        h = getattr(self.cfg, "CAGE_HALF", None)
        if not h:
            return ""
        if getattr(self.cfg, "CAGE_STYLE", "bedrock") == "barrier" and self._hm is not None:
            ring = sorted(set([(x, z) for x in range(-h, h + 1) for z in (-h, h)]
                              + [(x, z) for z in range(-h, h + 1) for x in (-h, h)]))
            out = ""
            for x, z in ring:
                fy = int(self._feet_y(x + 0.5, z + 0.5))
                out += ('    <DrawCuboid x1="{0}" y1="{1}" z1="{2}" x2="{0}" y2="{3}" z2="{2}" '
                        'type="barrier"/>\n').format(x, fy - 1, z, fy + 3)
            return out
        if getattr(self.cfg, "CAGE_STYLE", "bedrock") == "fence" and self._hm is not None:
            ring = sorted(set([(x, z) for x in range(-h, h + 1) for z in (-h, h)]
                              + [(x, z) for z in range(-h, h + 1) for x in (-h, h)]))
            return "".join(
                '    <DrawBlock x="{0}" y="{1}" z="{2}" type="fence"/>\n'.format(
                    x, int(self._feet_y(x + 0.5, z + 0.5)), z) for x, z in ring)
        y0, y1 = self.cfg.CAGE_FLOOR_Y, self.cfg.CAGE_TOP_Y
        boxes = [(-h, y0, -h, h, y1, -h), (-h, y0, h, h, y1, h),
                 (-h, y0, -h, -h, y1, h), (h, y0, -h, h, y1, h),
                 (-h, y0, -h, h, y0, h)]
        return "".join(
            '    <DrawCuboid x1="{0}" y1="{1}" z1="{2}" x2="{3}" y2="{4}" z2="{5}" '
            'type="bedrock"/>\n'.format(*b) for b in boxes)

    def step(self, action):
        self._steps += 1

        # Snapshot target state BEFORE acting/observing (for reward deltas)
        prev_dist    = self._tgt_dist
        prev_life    = self._tgt_life
        prev_visible = self._tgt_visible

        self._take_action(action)

        self._dmg_signal = 0.0
        obs_dict, world_state = self._get_obs_dict()
        obs = self._build_obs_vector(obs_dict)  # updates target + prev_pos state
        # One-hit kills (rabbit) can vanish before a life drop is ever observed,
        # so trust Malmo's own damage event (RewardForDamagingEntity) as "hit".
        if getattr(self.cfg, "USE_DAMAGE_SIGNAL", False) and self._dmg_signal > 0:
            self._damaged = True

        # Auto-fire: attack when aligned and in range (fires before next obs tick)
        if (getattr(self.cfg, 'AUTO_ATTACK', False)
                and self._in_range
                and abs(self._heading_error) < getattr(self.cfg, 'AUTO_ATTACK_ANGLE', 0.15)):
            self._agent_host.sendCommand("attack 1")
            time.sleep(0.05)
            self._agent_host.sendCommand("attack 0")

        reward, done, outcome = self._get_reward(obs_dict, prev_dist, prev_life, prev_visible)

        # Several prey: each kill pays REWARD_KILL; the episode continues
        # (re-targeting the nearest survivor) until KILLS_TO_END are dead.
        if outcome == "killed":
            self._kills += 1
            if self._kills < getattr(self.cfg, "KILLS_TO_END", 1):
                done, outcome = False, "kill"
                self._damaged, self._prev_phi, self._prev_phi_p = False, 0.0, 0.0
                self._prev_tgt_pos, self._reacquire = None, True

        # Mission ended unexpectedly
        if not world_state.is_mission_running and not done:
            done, outcome = True, "mission_ended"
            reward = self.cfg.REWARD_TIMEOUT

        # Timeout
        if self._steps >= self.cfg.MAX_STEPS and not done:
            done, outcome = True, "timeout"
            reward = self.cfg.REWARD_TIMEOUT

        if done:
            try:
                self._agent_host.sendCommand("quit")
            except Exception:
                pass

        info = {
            "outcome":    outcome,
            "steps":      self._steps,
            "pos":        (obs_dict.get("XPos", 0),
                           obs_dict.get("YPos", 0),
                           obs_dict.get("ZPos", 0)),
            "yaw":        float(obs_dict.get("Yaw",   0.0)),
            "pitch":      float(obs_dict.get("Pitch", 0.0)),
            "on_ground":  int(obs_dict.get("OnGround", False)),
            "action":     self.actions[action][0],
            "target_dist": float(self._tgt_dist),
            "target_life": float(self._tgt_life),
            "killed":     int(outcome == "killed"),
            "kills":      self._kills,
        }
        # Scripted teachers / feature probes only — never part of the model's obs.
        if getattr(self.cfg, "EMIT_PRIVILEGED", False) and self._last_target is not None:
            info["privileged"] = [round(float(v), 5) for v in self._last_target]
        return obs, reward, done, info

    def close(self):
        try:
            self._agent_host.sendCommand("quit")
        except Exception:
            pass

    # ── Target spawn placement (the predictability knob) ─────────────────────
    def _choose_target_spawn(self):
        sx, _, sz = self.cfg.SPAWN
        mode = self.cfg.TARGET_MODE
        if mode == "penned":
            return sx, sz + self.cfg.PENNED_DZ
        lo, hi = self.cfg.ARENA_MIN, self.cfg.ARENA_MAX
        if self._hm is not None:
            # Only standable ground columns (never a treetop) inside the arena.
            far = self.cfg.FLEEING_MIN_DIST if mode == "fleeing" else 0.0
            cand = [(x + 0.5, z + 0.5) for x, z in self._hm_ok
                    if lo <= x <= hi and lo <= z <= hi
                    and math.hypot(x + 0.5 - sx, z + 0.5 - sz) >= far]
            return random.choice(cand)
        if mode == "fleeing":
            for _ in range(20):
                tx = random.uniform(lo, hi)
                tz = random.uniform(lo, hi)
                if math.hypot(tx - sx, tz - sz) >= self.cfg.FLEEING_MIN_DIST:
                    return tx, tz
            return tx, tz
        # wandering (default fallback)
        return random.uniform(lo, hi), random.uniform(lo, hi)

    # ── Action execution ─────────────────────────────────────────────────────
    def _take_action(self, action_idx):
        name, cmds_on, cmds_off = self.actions[action_idx]
        # Cancel residual camera velocity from Minecraft's native mouse look
        self._agent_host.sendCommand("pitch 0")
        self._agent_host.sendCommand("turn 0")
        for cmd in cmds_on:
            self._agent_host.sendCommand(cmd)
        time.sleep(self.cfg.STEP_DURATION)
        for cmd in cmds_off:
            self._agent_host.sendCommand(cmd)

    # ── Observation building ─────────────────────────────────────────────────
    def _build_obs_vector(self, obs):
        x   = float(obs.get("XPos",  self.cfg.SPAWN[0]))
        y   = float(obs.get("YPos",  self.cfg.SPAWN[1]))
        z   = float(obs.get("ZPos",  self.cfg.SPAWN[2]))
        yaw = float(obs.get("Yaw",   0.0))
        pit = float(obs.get("Pitch", 0.0))
        gnd = float(obs.get("OnGround", False))

        pos            = np.array([x, y, z], dtype=np.float32)
        vel            = pos - self._prev_pos
        self._prev_pos = pos

        food   = float(obs.get("Food", 20.0)) / 20.0
        health = float(obs.get("Life", 20.0)) / 20.0

        proprio = np.array([
            gnd, yaw / 180.0, pit / 90.0,
            vel[1], vel[0], vel[2],
            x - self.cfg.SPAWN[0], y - self.cfg.SPAWN[1], z - self.cfg.SPAWN[2],
            food, health,
        ], dtype=np.float32)

        target = self._target_features(obs, x, y, z, yaw)
        self._last_target = target

        # Pixels + proprioception: the model gets no target/voxel stream (the
        # target state above still drives reward and info["privileged"]).
        if getattr(self.cfg, "OBS_MODE", "full") == "proprio":
            return proprio

        raw_grid = obs.get("floor3x3", [])
        voxels   = self._encode_grid(raw_grid)

        return np.concatenate([proprio, target, voxels])

    def _target_features(self, obs, x, y, z, yaw):
        """Compute the 12-dim moving-target stream and update target state."""
        entities = obs.get("entities", [])
        self._n_targets_prev = self._n_targets
        self._n_targets = sum(1 for e in entities if e.get("name") == self.cfg.TARGET_MOB)
        ent = self._find_target(entities, x, z)

        if ent is not None:
            tx, ty, tz = float(ent.get("x", x)), float(ent.get("y", y)), float(ent.get("z", z))
            tgt_pos = np.array([tx, ty, tz], dtype=np.float32)
            if self._prev_tgt_pos is None:
                tgt_vel = np.zeros(3, dtype=np.float32)
            else:
                tgt_vel = tgt_pos - self._prev_tgt_pos
            self._prev_tgt_pos = tgt_pos
            self._tgt_pos      = tgt_pos

            dx, dy, dz = tx - x, ty - y, tz - z
            dist = float(math.sqrt(dx * dx + dy * dy + dz * dz))
            self._tgt_dist = dist

            # Heading error: Malmo yaw 0 faces +Z (south). desired = -atan2(dx, dz).
            desired_yaw = -180.0 * math.atan2(dx, dz) / math.pi
            herr = desired_yaw - yaw
            while herr < -180.0:
                herr += 360.0
            while herr > 180.0:
                herr -= 360.0
            heading_error = herr / 180.0

            # Pitch error: aim the crosshair at the body middle (eye at feet+1.62).
            pit = float(obs.get("Pitch", 0.0))
            horiz = math.hypot(dx, dz)
            want_pitch = -180.0 * math.atan2((dy + self.cfg.TARGET_HEIGHT / 2.0) - 1.62, max(horiz, 0.3)) / math.pi
            self._pitch_deg   = pit
            self._pitch_error = max(-2.0, min(2.0, (want_pitch - pit) / 90.0))

            life = float(ent.get("life", self.cfg.TARGET_MAX_LIFE))
            self._tgt_life = life
            self._tgt_visible = True
            self._target_seen = True

            in_range = 1.0 if dist <= self.cfg.ATTACK_RANGE else 0.0
            los_hit  = 1.0 if self._los_on_target(obs) else 0.0
            self._los_hit       = bool(los_hit)
            self._heading_error = heading_error
            self._in_range      = bool(in_range)

            return np.array([
                dx, dy, dz,
                tgt_vel[0], tgt_vel[1], tgt_vel[2],
                dist / self.cfg.DIST_SCALE,
                heading_error,
                los_hit, in_range,
                life / self.cfg.TARGET_MAX_LIFE,
                1.0,
            ], dtype=np.float32)

        # Target not currently visible
        self._tgt_visible   = False
        self._pitch_error   = 0.0
        self._pitch_deg     = float(obs.get("Pitch", 0.0))
        self._los_hit       = False
        self._heading_error = 0.0
        self._in_range      = False
        return np.array([
            0, 0, 0, 0, 0, 0,
            1.0,            # max normalized distance
            0,              # heading error unknown
            0, 0,
            self._tgt_life / self.cfg.TARGET_MAX_LIFE,
            0.0,            # not visible
        ], dtype=np.float32)

    def _find_target(self, entities, ax=0.0, az=0.0):
        """Return the target-mob entity nearest the agent, or None."""
        best, best_d = None, 1e9
        for ent in entities:
            if ent.get("name") != self.cfg.TARGET_MOB:
                continue
            d = abs(float(ent.get("x", 0)) - ax) + abs(float(ent.get("z", 0)) - az)
            if d < best_d:
                best, best_d = ent, d
        return best

    def _los_on_target(self, obs):
        los = obs.get("LineOfSight", {})
        if not los:
            return False
        return los.get("type") == self.cfg.TARGET_MOB or los.get("hitType") == "entity"

    def _encode_grid(self, raw_grid):
        expected = self.cfg.GRID_SIZE
        if len(raw_grid) != expected:
            return np.zeros(expected, dtype=np.float32)
        encoded = np.zeros(expected, dtype=np.float32)
        for i, block in enumerate(raw_grid):
            encoded[i] = float(self.cfg.BLOCK_ENCODING.get(block, 1))
        return encoded

    # ── Reward / termination ─────────────────────────────────────────────────
    def _get_reward(self, obs, prev_dist, prev_life, prev_visible):
        y = float(obs.get("YPos", self.cfg.SPAWN[1]))

        # Fell out of the world (defensive — arena is walled/flat)
        if y < self.cfg.FALL_Y_THRESHOLD:
            return self.cfg.REWARD_FELL, True, "fell"

        # Stuck in a pit (e.g. shoved in by a pig): no way to hunt from there, so
        # charge for the time and end the episode if it does not climb out.
        pit_depth = getattr(self.cfg, "PIT_DEPTH", 0.0)
        if pit_depth and getattr(self, "_hm", None) is not None:
            x, z = float(obs.get("XPos", 0.0)), float(obs.get("ZPos", 0.0))
            if self._pit_depth(x, z, y) >= pit_depth:
                self._pit_steps += 1
                if self._pit_steps >= self.cfg.PIT_MAX_STEPS:
                    return self.cfg.REWARD_PIT, True, "stuck_in_pit"
            else:
                self._pit_steps = 0

        # With the damage signal, only the agent's own hits count: the prey can
        # also lose health to the world (falls, drowning, suffocation).
        dmg_sig = getattr(self.cfg, "USE_DAMAGE_SIGNAL", False)

        # Kill: target visible with no life left, OR vanished after being damaged
        if self._tgt_visible and self._tgt_life <= 0.0 and (self._damaged or not dmg_sig):
            return self.cfg.REWARD_KILL, True, "killed"
        if self._target_seen and prev_visible and not self._tgt_visible and self._damaged:
            return self.cfg.REWARD_KILL, True, "killed"
        # Several targets (or wild ones nearby): the nearest one can change, so
        # count bodies instead.
        if ((getattr(self.cfg, "N_TARGETS", 1) > 1 or getattr(self.cfg, "USE_DAMAGE_SIGNAL", False))
                and self._damaged
                and self._n_targets < getattr(self, "_n_targets_prev", 0)):
            return self.cfg.REWARD_KILL, True, "killed"

        reward = self.cfg.REWARD_STEP_PENALTY
        if self._pit_steps:
            reward -= getattr(self.cfg, "PIT_STEP_PENALTY", 0.0)

        if prev_visible and self._tgt_visible:
            # Hit reward: target lost health since last step
            if self._tgt_life < prev_life and (not dmg_sig or self._dmg_signal > 0):
                reward += self.cfg.REWARD_HIT * (prev_life - self._tgt_life)
                self._damaged = True
            # Approach shaping: reward closing distance (clamped). Skipped on
            # the step after a kill, when the target jumps to the next cow.
            delta = 0.0 if self._reacquire else prev_dist - self._tgt_dist
            delta = max(-self.cfg.APPROACH_CLIP, min(self.cfg.APPROACH_CLIP, delta))
            reward += self.cfg.REWARD_APPROACH_COEF * delta
            # Aim bonus: crosshair on the animal
            if self._los_hit:
                reward += self.cfg.REWARD_AIM
            # Potential-based aim-alignment shaping (Ng 1999, policy-invariant).
            # Φ(s) = cos(heading_error · π): 1 when perfectly aimed, -1 when opposite.
            # Shaping = AIM_ALIGN_COEF · (γ·Φ' − Φ) — adds zero net discounted reward.
            align_coef = getattr(self.cfg, 'AIM_ALIGN_COEF', 0.0)
            if align_coef:
                phi_new = math.cos(self._heading_error * math.pi)
                reward += align_coef * (self.cfg.GAMMA * phi_new - self._prev_phi)
                self._prev_phi = phi_new
            # Pitch counterpart: pulls the crosshair up/down onto the body.
            pitch_coef = getattr(self.cfg, 'PITCH_ALIGN_COEF', 0.0)
            if pitch_coef:
                phi_p = math.cos(min(abs(self._pitch_error), 1.0) * math.pi / 2.0)
                reward += pitch_coef * (self.cfg.GAMMA * phi_p - self._prev_phi_p)
                self._prev_phi_p = phi_p
        else:
            self._prev_phi = 0.0
            self._prev_phi_p = 0.0

        # Staring at the sky/ground helps with nothing: small penalty past the limit.
        lim = getattr(self.cfg, 'PITCH_LIMIT', 0.0)
        if lim and abs(self._pitch_deg) > lim:
            reward -= getattr(self.cfg, 'PITCH_PENALTY', 0.0)

        self._reacquire = False
        return reward, False, "alive"

    # ── Malmo interaction (mirrors BridgingEnv / ParkourEnv) ─────────────────
    def _start_mission(self, mission_xml, max_retries=3,
                       begin_timeout=60.0, running_timeout=30.0):
        t0 = time.time()
        ws = self._agent_host.getWorldState()
        while ws.is_mission_running:
            if time.time() - t0 > running_timeout:
                print("\nWARNING: Previous mission still running after {0}s, "
                      "forcing quit".format(running_timeout))
                try:
                    self._agent_host.sendCommand("quit")
                except Exception:
                    pass
                time.sleep(1)
                break
            time.sleep(0.1)
            ws = self._agent_host.getWorldState()
        time.sleep(0.5)

        mission        = MalmoPython.MissionSpec(mission_xml, True)
        if self._video:
            from envs.video_record import request_video
            request_video(mission, self.cfg)
        mission_record = MalmoPython.MissionRecordSpec()
        client_pool    = MalmoPython.ClientPool()
        client_pool.add(MalmoPython.ClientInfo("127.0.0.1", self._malmo_port))

        for attempt in range(max_retries):
            try:
                self._agent_host.startMission(mission, client_pool, mission_record, 0, "")
            except RuntimeError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Could not start mission after {0} attempts: {1}\n"
                        "Is Minecraft running on port {2}?".format(
                            max_retries, e, self._malmo_port))
                print("  Retrying start ({0}/{1})...".format(attempt + 1, max_retries))
                time.sleep(2)
                continue

            print("Waiting for mission to start...", end=" ")
            t0 = time.time()
            ws = self._agent_host.getWorldState()
            timed_out = False
            while not ws.has_mission_begun:
                if time.time() - t0 > begin_timeout:
                    timed_out = True
                    break
                print(".", end="", flush=True)
                time.sleep(0.1)
                ws = self._agent_host.getWorldState()
                for error in ws.errors:
                    print("\nMission error:", error.text)

            if not timed_out:
                print(" ready!")
                time.sleep(0.5)
                return

            print("\nWARNING: Mission did not begin within {0}s".format(begin_timeout))
            if attempt < max_retries - 1:
                print("  Retrying ({0}/{1})...".format(attempt + 1, max_retries))
                time.sleep(2)

        raise RuntimeError(
            "Mission failed to begin after {0} attempts "
            "(port {1})".format(max_retries, self._malmo_port))

    def _get_obs_dict(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            ws = self._agent_host.getWorldState()
            if getattr(self.cfg, "USE_DAMAGE_SIGNAL", False):
                for r in ws.rewards:
                    self._dmg_signal += r.getValue()
            # getWorldState() clears the "since last state" frame counter, so a
            # frame must be harvested from THIS ws, not a later poll.
            if self._video and len(ws.video_frames) > 0:
                f = ws.video_frames[-1]
                self.last_frame = np.frombuffer(
                    bytes(f.pixels), dtype=np.uint8
                ).reshape(f.height, f.width, self.cfg.VIDEO_CHANNELS).copy()
            if ws.number_of_observations_since_last_state > 0:
                return json.loads(ws.observations[-1].text), ws
            time.sleep(0.03)
        return {}, self._agent_host.getWorldState()

    def _get_observation(self):
        obs_dict, _ = self._get_obs_dict()
        return self._build_obs_vector(obs_dict)
