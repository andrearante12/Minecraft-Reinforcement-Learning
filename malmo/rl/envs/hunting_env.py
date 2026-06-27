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
        self.cfg       = cfg
        self.actions   = cfg.ACTIONS
        self.n_actions = cfg.N_ACTIONS
        self._malmo_port = malmo_port if malmo_port is not None else cfg.MALMO_PORT

        self._prev_pos = np.array(cfg.SPAWN, dtype=np.float32)
        self._steps    = 0

        # Target-tracking state (updated each step in _build_obs_vector)
        self._prev_tgt_pos = None
        self._tgt_pos      = None
        self._tgt_dist     = float(cfg.DIST_SCALE)
        self._tgt_life     = cfg.TARGET_MAX_LIFE
        self._tgt_visible  = False
        self._los_hit      = False
        self._target_seen  = False     # ever seen the animal this episode
        self._damaged      = False     # ever dealt damage this episode

        self.observation_shape = (cfg.INPUT_SIZE,)

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
        self._prev_pos     = np.array(self.cfg.SPAWN, dtype=np.float32)
        self._prev_tgt_pos = None
        self._tgt_pos      = None
        self._tgt_dist     = float(self.cfg.DIST_SCALE)
        self._tgt_life     = self.cfg.TARGET_MAX_LIFE
        self._tgt_visible  = False
        self._los_hit      = False
        self._target_seen  = False
        self._damaged      = False

        tx, tz = self._choose_target_spawn()
        draw_entity = '    <DrawEntity x="{x:.1f}" y="46.0" z="{z:.1f}" type="{mob}"/>\n'.format(
            x=tx, z=tz, mob=self.cfg.TARGET_MOB)
        mission_xml = self._mission_xml_base.replace(
            '</DrawingDecorator>', draw_entity + '      </DrawingDecorator>')

        time.sleep(0.5)
        self._start_mission(mission_xml)
        print("  [reset] target {mob} at ({x:.1f}, {z:.1f})  mode={mode}".format(
            mob=self.cfg.TARGET_MOB, x=tx, z=tz, mode=self.cfg.TARGET_MODE))
        return self._get_observation()

    def step(self, action):
        self._steps += 1

        # Snapshot target state BEFORE acting/observing (for reward deltas)
        prev_dist    = self._tgt_dist
        prev_life    = self._tgt_life
        prev_visible = self._tgt_visible

        self._take_action(action)

        obs_dict, world_state = self._get_obs_dict()
        obs = self._build_obs_vector(obs_dict)  # updates target + prev_pos state

        reward, done, outcome = self._get_reward(obs_dict, prev_dist, prev_life, prev_visible)

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
        }
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

        raw_grid = obs.get("floor3x3", [])
        voxels   = self._encode_grid(raw_grid)

        return np.concatenate([proprio, target, voxels])

    def _target_features(self, obs, x, y, z, yaw):
        """Compute the 12-dim moving-target stream and update target state."""
        ent = self._find_target(obs.get("entities", []))

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

            life = float(ent.get("life", self.cfg.TARGET_MAX_LIFE))
            self._tgt_life = life
            self._tgt_visible = True
            self._target_seen = True

            in_range = 1.0 if dist <= self.cfg.ATTACK_RANGE else 0.0
            los_hit  = 1.0 if self._los_on_target(obs) else 0.0
            self._los_hit = bool(los_hit)

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
        self._tgt_visible = False
        self._los_hit     = False
        return np.array([
            0, 0, 0, 0, 0, 0,
            1.0,            # max normalized distance
            0,              # heading error unknown
            0, 0,
            self._tgt_life / self.cfg.TARGET_MAX_LIFE,
            0.0,            # not visible
        ], dtype=np.float32)

    def _find_target(self, entities):
        """Return the nearest entity matching the target mob, or None."""
        best, best_d = None, 1e9
        for ent in entities:
            if ent.get("name") != self.cfg.TARGET_MOB:
                continue
            d = abs(float(ent.get("x", 0))) + abs(float(ent.get("z", 0)))
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

        # Kill: target visible with no life left, OR vanished after being damaged
        if self._tgt_visible and self._tgt_life <= 0.0:
            return self.cfg.REWARD_KILL, True, "killed"
        if self._target_seen and prev_visible and not self._tgt_visible and self._damaged:
            return self.cfg.REWARD_KILL, True, "killed"

        reward = self.cfg.REWARD_STEP_PENALTY

        if prev_visible and self._tgt_visible:
            # Hit reward: target lost health since last step
            if self._tgt_life < prev_life:
                reward += self.cfg.REWARD_HIT * (prev_life - self._tgt_life)
                self._damaged = True
            # Approach shaping: reward closing distance (clamped)
            delta = prev_dist - self._tgt_dist
            delta = max(-self.cfg.APPROACH_CLIP, min(self.cfg.APPROACH_CLIP, delta))
            reward += self.cfg.REWARD_APPROACH_COEF * delta
            # Aim bonus: crosshair on the animal
            if self._los_hit:
                reward += self.cfg.REWARD_AIM

        return reward, False, "alive"

    # ── Malmo interaction (mirrors BridgingEnv / ParkourEnv) ─────────────────
    def _start_mission(self, mission_xml, max_retries=3,
                       begin_timeout=30.0, running_timeout=30.0):
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
            if ws.number_of_observations_since_last_state > 0:
                return json.loads(ws.observations[-1].text), ws
            time.sleep(0.03)
        return {}, self._agent_host.getWorldState()

    def _get_observation(self):
        obs_dict, _ = self._get_obs_dict()
        return self._build_obs_vector(obs_dict)
