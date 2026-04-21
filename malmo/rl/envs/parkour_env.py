"""
envs/parkour_env.py
-------------------
Unified Gym-style Malmo wrapper for all parkour environments.
Behavioral differences between environments are driven entirely by config flags.

Interface:
    env = ParkourEnv(cfg)
    obs = env.reset()                          # np.ndarray (INPUT_SIZE,)
    obs, reward, done, info = env.step(action) # action is an int 0-14

Observation vector layout (INPUT_SIZE = 159):
    [0]      onGround        (0 or 1)
    [1]      yaw             (normalized -1 to 1)
    [2]      pitch           (normalized -1 to 1)
    [3]      delta_y         (velocity inferred from position diff)
    [4]      delta_x
    [5]      delta_z
    [6]      goal_dx         (goal_x - agent_x)
    [7]      goal_dy
    [8]      goal_dz
    [9:159]  voxel grid      (5 x 5 x 6 = 150 values, encoded as ints)
"""

import sys
import os
import time
import json
import numpy as np

# ── Add parkour/ root to path so training.configs is importable ───────────────
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


class ParkourEnv:
    def __init__(self, cfg, malmo_port=None, force_reset=False):
        self.cfg       = cfg
        self.actions   = cfg.ACTIONS
        self.n_actions = cfg.N_ACTIONS
        self._malmo_port = malmo_port if malmo_port is not None else cfg.MALMO_PORT

        self._prev_pos = np.array(cfg.SPAWN, dtype=np.float32)
        self._goal_pos = np.array(cfg.GOAL_POS, dtype=np.float32)
        self._steps    = 0
        self._landing_counter = 0
        self._landing_active  = False
        self._jump_checkpoints    = list(getattr(cfg, 'JUMP_CHECKPOINTS', []))
        self._next_checkpoint_idx = 0

        self.observation_shape = (cfg.INPUT_SIZE,)

        with open(cfg.MISSION_FILE, "r") as f:
            xml = f.read()
        # Store both variants: force_reset for first load after env switch,
        # no_force for subsequent same-env resets (much faster)
        self._mission_xml_force = xml
        self._mission_xml_fast  = xml.replace('forceReset="true"', 'forceReset="false"')
        self._next_force_reset  = force_reset

        self._agent_host = MalmoPython.AgentHost()
        try:
            self._agent_host.parse(["env_server"])
        except RuntimeError as e:
            print("ERROR parsing AgentHost:", e)
            sys.exit(1)

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        self._steps    = 0
        self._prev_pos = np.array(self.cfg.SPAWN, dtype=np.float32)
        self._landing_counter = 0
        self._landing_active  = False
        self._next_checkpoint_idx = 0
        time.sleep(0.5)
        # Use forceReset only on first reset after an env switch, then fast resets
        if self._next_force_reset:
            self._start_mission(self._mission_xml_force)
            self._next_force_reset = False
        else:
            self._start_mission(self._mission_xml_fast)
        return self._get_observation()

    def step(self, action):
        self._steps += 1
        prev_pos = self._prev_pos.copy()

        self._take_action(action)

        obs_dict, world_state = self._get_obs_dict()
        obs          = self._build_obs_vector(obs_dict)  # updates self._prev_pos
        reward, done, outcome = self._get_reward(obs_dict, prev_pos)

        if not world_state.is_mission_running and not done:
            done, outcome = True, "mission_ended"
            if self.cfg.REWARD_ON_MISSION_ENDED:
                reward = self.cfg.REWARD_TIMEOUT
                if self.cfg.PROXIMITY_SCALED_TERMINAL:
                    x_now = float(obs_dict.get("XPos", self.cfg.SPAWN[0]))
                    y_now = float(obs_dict.get("YPos", self.cfg.SPAWN[1]))
                    z_now = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))
                    reward *= (1.0 - self._proximity(np.array([x_now, y_now, z_now])))
        if self._steps >= self.cfg.MAX_STEPS and not done:
            z_now = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))
            y_now = float(obs_dict.get("YPos", self.cfg.SPAWN[1]))
            nm_y_min = self.cfg.Y_SUCCESS_MIN if self.cfg.Y_SUCCESS_MIN is not None else self.cfg.FALL_Y_THRESHOLD
            nm_x_ok = True
            if self.cfg.X_SUCCESS_MIN is not None:
                x_now = float(obs_dict.get("XPos", self.cfg.SPAWN[0]))
                if x_now < self.cfg.X_SUCCESS_MIN - self.cfg.NEAR_MISS_THRESHOLD:
                    nm_x_ok = False
                if self.cfg.X_SUCCESS_MAX is not None and x_now > self.cfg.X_SUCCESS_MAX + self.cfg.NEAR_MISS_THRESHOLD:
                    nm_x_ok = False
            if (z_now >= (self.cfg.Z_SUCCESS - self.cfg.NEAR_MISS_THRESHOLD)
                    and y_now >= nm_y_min and nm_x_ok):
                reward = self.cfg.REWARD_NEAR_MISS
                done, outcome = True, "near_miss"
            else:
                done, outcome = True, "timeout"
                reward = self.cfg.REWARD_TIMEOUT
                if self.cfg.PROXIMITY_SCALED_TERMINAL:
                    x_now = float(obs_dict.get("XPos", self.cfg.SPAWN[0]))
                    reward *= (1.0 - self._proximity(np.array([x_now, y_now, z_now])))

        if done:
            try:
                self._agent_host.sendCommand("quit")
            except Exception:
                pass

        # Advance jump checkpoints: agent must be alive (above fall threshold) and past each z
        if self._jump_checkpoints:
            z_now = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))
            y_now = float(obs_dict.get("YPos", self.cfg.SPAWN[1]))
            while self._next_checkpoint_idx < len(self._jump_checkpoints):
                if (y_now >= self.cfg.FALL_Y_THRESHOLD
                        and z_now >= self._jump_checkpoints[self._next_checkpoint_idx]):
                    self._next_checkpoint_idx += 1
                else:
                    break

        info = {
            "outcome":         outcome,
            "steps":           self._steps,
            "pos":             (obs_dict.get("XPos", 0),
                                obs_dict.get("YPos", 0),
                                obs_dict.get("ZPos", 0)),
            "yaw":             obs_dict.get("Yaw",      0.0),
            "pitch":           obs_dict.get("Pitch",    0.0),
            "on_ground":       int(obs_dict.get("OnGround", True)),
            "action":          self.actions[action][0],
            "jumps_completed": self._next_checkpoint_idx,
        }
        return obs, reward, done, info

    def close(self):
        try:
            self._agent_host.sendCommand("quit")
        except Exception:
            pass

    # ── Malmo interaction ──────────────────────────────────────────────────────

    def _start_mission(self, mission_xml=None, max_retries=5,
                       begin_timeout=45.0, running_timeout=30.0):
        # Wait for any previous mission to finish
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
                time.sleep(3)
                break
            time.sleep(0.1)
            ws = self._agent_host.getWorldState()
        time.sleep(0.5)

        if mission_xml is None:
            mission_xml = self._mission_xml_fast
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
                time.sleep(5)
                continue

            # Wait for the mission to actually begin (with timeout)
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

            # Mission begin timed out — send quit to clean up Minecraft state before retrying
            print("\nWARNING: Mission did not begin within {0}s".format(begin_timeout))
            try:
                self._agent_host.sendCommand("quit")
            except Exception:
                pass
            if attempt < max_retries - 1:
                backoff = 5 * (attempt + 1)
                print("  Retrying ({0}/{1}) after {2}s...".format(attempt + 1, max_retries, backoff))
                time.sleep(backoff)

        raise RuntimeError(
            "Mission failed to begin after {0} attempts "
            "(port {1})".format(max_retries, self._malmo_port))

    def _take_action(self, action_idx):
        _, cmds_on, cmds_off = self.actions[action_idx]
        for cmd in cmds_on:
            self._agent_host.sendCommand(cmd)
        time.sleep(self.cfg.STEP_DURATION)
        for cmd in cmds_off:
            self._agent_host.sendCommand(cmd)

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

    # ── Observation building ───────────────────────────────────────────────────

    def _build_obs_vector(self, obs):
        x   = float(obs.get("XPos",     self.cfg.SPAWN[0]))
        y   = float(obs.get("YPos",     self.cfg.SPAWN[1]))
        z   = float(obs.get("ZPos",     self.cfg.SPAWN[2]))
        yaw = float(obs.get("Yaw",      0.0))
        pit = float(obs.get("Pitch",    0.0))
        gnd = float(obs.get("OnGround", False))

        pos            = np.array([x, y, z], dtype=np.float32)
        prev_pos       = self._prev_pos.copy()
        vel            = pos - prev_pos
        self._prev_pos = pos

        proprio = np.array([
            gnd,
            yaw / 180.0,
            pit / 90.0,
            vel[1],
            vel[0],
            vel[2],
        ], dtype=np.float32)

        goal_delta = (self._goal_pos - pos).astype(np.float32)

        raw_grid = obs.get("floor3x3", [])
        voxels   = self._encode_grid(raw_grid)

        return np.concatenate([proprio, goal_delta, voxels])

    def _encode_grid(self, raw_grid):
        expected = self.cfg.GRID_SIZE
        if len(raw_grid) != expected:
            return np.zeros(expected, dtype=np.float32)
        encoded = np.zeros(expected, dtype=np.float32)
        for i, block in enumerate(raw_grid):
            encoded[i] = float(self.cfg.BLOCK_ENCODING.get(block, 1))
        return encoded

    # ── Reward function ────────────────────────────────────────────────────────

    def _proximity(self, pos):
        """Return 0.0 (at spawn) to 1.0 (at goal) based on distance to goal."""
        initial_dist = np.linalg.norm(
            np.array(self.cfg.SPAWN, dtype=np.float32) - np.array(self.cfg.GOAL_POS, dtype=np.float32)
        )
        current_dist = np.linalg.norm(pos - np.array(self.cfg.GOAL_POS, dtype=np.float32))
        if initial_dist == 0:
            return 1.0
        return max(1.0 - current_dist / initial_dist, 0.0)

    def _in_landing_zone(self, x, y, z):
        """Check if position is within the landing zone bounds."""
        if z < self.cfg.Z_SUCCESS:
            return False
        if self.cfg.Z_SUCCESS_MAX is not None and z > self.cfg.Z_SUCCESS_MAX:
            return False
        y_min = self.cfg.Y_SUCCESS_MIN if self.cfg.Y_SUCCESS_MIN is not None else self.cfg.FALL_Y_THRESHOLD
        if y < y_min:
            return False
        if self.cfg.X_SUCCESS_MIN is not None and x < self.cfg.X_SUCCESS_MIN:
            return False
        if self.cfg.X_SUCCESS_MAX is not None and x > self.cfg.X_SUCCESS_MAX:
            return False
        return True

    def _get_reward(self, obs, prev_pos):
        x = float(obs.get("XPos", self.cfg.SPAWN[0]))
        y = float(obs.get("YPos", self.cfg.SPAWN[1]))
        z = float(obs.get("ZPos", self.cfg.SPAWN[2]))

        if y < self.cfg.FALL_Y_THRESHOLD:
            reward = self.cfg.REWARD_FELL
            if self.cfg.PROXIMITY_SCALED_TERMINAL:
                reward *= (1.0 - self._proximity(np.array([x, y, z])))
            # Landing phase interrupted by falling
            self._landing_active = False
            self._landing_counter = 0
            return reward, True, "fell"

        # ── Landing phase logic ──────────────────────────────────────────────
        if self._landing_active:
            if self._in_landing_zone(x, y, z):
                self._landing_counter += 1
                if self._landing_counter >= self.cfg.LANDING_TICKS:
                    return self.cfg.REWARD_SUCCESS, True, "landed"
                return self.cfg.REWARD_LANDING_TICK, False, "alive"
            else:
                # Fell off / overshot — full fell penalty (no proximity scaling,
                # the agent WAS on the block and failed to stay)
                self._landing_active = False
                self._landing_counter = 0
                return self.cfg.REWARD_FELL, True, "fell"

        # ── Success check ────────────────────────────────────────────────────
        if self._in_landing_zone(x, y, z):
            if self.cfg.LANDING_TICKS > 0:
                # Enter landing phase — no reward yet, must survive a tick first
                self._landing_active = True
                self._landing_counter = 0
                return 0.0, False, "alive"
            else:
                # Instant success (backward compat)
                return self.cfg.REWARD_SUCCESS, True, "landed"

        pos_now   = np.array([x, y, z], dtype=np.float32)
        dist_now  = np.linalg.norm(pos_now - self._goal_pos)
        dist_prev = np.linalg.norm(prev_pos - self._goal_pos)
        progress  = dist_prev - dist_now
        reward    = (self.cfg.REWARD_STEP_PENALTY
                     + self.cfg.REWARD_PROGRESS_COEF * progress)
        return reward, False, "alive"
