"""
envs/simple_jump/env.py
--------------------------------
Gym-style Malmo wrapper for the simple jump environment.

Interface:
    env = ParkourEnv(cfg)
    obs = env.reset()                          # np.ndarray (INPUT_SIZE,)
    obs, reward, done, info = env.step(action) # action is an int 0-11

Observation vector layout (INPUT_SIZE = 129):
    [0]      onGround        (0 or 1)
    [1]      yaw             (normalized -1 to 1)
    [2]      pitch           (normalized -1 to 1)
    [3]      delta_y         (velocity inferred from position diff)
    [4]      delta_x
    [5]      delta_z
    [6]      goal_dx         (goal_x - agent_x)
    [7]      goal_dy
    [8]      goal_dz
    [9:129]  voxel grid      (5 x 4 x 6 = 120 values, encoded as ints)
"""

import sys
import os
import time
import json
import numpy as np

# ── Add parkour/ root to path so training.configs is importable ───────────────
# This file lives at parkour/envs/simple_jump/env.py
# Three levels up reaches parkour/
PARKOUR_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, PARKOUR_ROOT)

# ── Config import ─────────────────────────────────────────────────────────────
from training.configs.simple_jump_cfg import SimpleJumpCFG as CFG

# ── Malmo import ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(CFG.MALMO_PYTHON))

try:
    import MalmoPython
except ImportError:
    raise ImportError(
        "Could not import MalmoPython.\n"
        "Tried: {0}".format(os.path.abspath(CFG.MALMO_PYTHON))
    )


class ParkourEnv:
    def __init__(self, cfg=CFG, malmo_port=None):
        self.cfg       = cfg
        self.actions   = cfg.ACTIONS
        self.n_actions = cfg.N_ACTIONS
        self._malmo_port = malmo_port if malmo_port is not None else cfg.MALMO_PORT

        self._prev_pos = np.array(cfg.SPAWN, dtype=np.float32)
        self._goal_pos = np.array(cfg.GOAL_POS, dtype=np.float32)
        self._prev_z   = float(cfg.SPAWN[2])
        self._steps    = 0

        self.observation_shape = (cfg.INPUT_SIZE,)

        with open(cfg.MISSION_FILE, "r") as f:
            xml = f.read()
        xml = xml.replace('forceReset="true"', 'forceReset="false"')
        self._mission_xml = xml

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
        self._prev_z   = float(self.cfg.SPAWN[2])
        time.sleep(0.5)
        self._start_mission()
        return self._get_observation()

    def step(self, action):
        self._steps += 1
        prev_z = self._prev_z

        self._take_action(action)

        obs_dict, world_state = self._get_obs_dict()
        self._prev_z = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))
        obs          = self._build_obs_vector(obs_dict)
        reward, done, outcome = self._get_reward(obs_dict, prev_z)

        if not world_state.is_mission_running and not done:
            done, outcome = True, "mission_ended"
            reward = self.cfg.REWARD_TIMEOUT   
        if self._steps >= self.cfg.MAX_STEPS:
            done, outcome = True, "timeout"
            reward = self.cfg.REWARD_TIMEOUT

        info = {
            "outcome": outcome,
            "steps":   self._steps,
            "pos":     (obs_dict.get("XPos", 0),
                        obs_dict.get("YPos", 0),
                        obs_dict.get("ZPos", 0)),
            "action":  self.actions[action][0],
        }
        return obs, reward, done, info

    def close(self):
        try:
            self._agent_host.sendCommand("quit")
        except Exception:
            pass

    # ── Malmo interaction ──────────────────────────────────────────────────────

    def _start_mission(self, max_retries=3):
        ws = self._agent_host.getWorldState()
        while ws.is_mission_running:
            time.sleep(0.1)
            ws = self._agent_host.getWorldState()
        time.sleep(0.5)

        mission        = MalmoPython.MissionSpec(self._mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()
        client_pool    = MalmoPython.ClientPool()
        client_pool.add(MalmoPython.ClientInfo("127.0.0.1", self._malmo_port))

        for attempt in range(max_retries):
            try:
                self._agent_host.startMission(mission, client_pool, mission_record, 0, "")
                break
            except RuntimeError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Could not start mission after {0} attempts: {1}\n"
                        "Is Minecraft running on port {2}?".format(
                            max_retries, e, self._malmo_port))
                print("  Retrying ({0}/{1})...".format(attempt + 1, max_retries))
                time.sleep(2)

        print("Waiting for mission to start...", end=" ")
        ws = self._agent_host.getWorldState()
        while not ws.has_mission_begun:
            print(".", end="", flush=True)
            time.sleep(0.1)
            ws = self._agent_host.getWorldState()
            for error in ws.errors:
                print("\nMission error:", error.text)
        print(" ready!")
        time.sleep(0.5)

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

    def _get_reward(self, obs, prev_z):
        y = float(obs.get("YPos", self.cfg.SPAWN[1]))
        z = float(obs.get("ZPos", self.cfg.SPAWN[2]))

        if y < self.cfg.FALL_Y_THRESHOLD:
            return self.cfg.REWARD_FELL, True, "fell"

        if z >= self.cfg.Z_SUCCESS and y >= self.cfg.FALL_Y_THRESHOLD:
            return self.cfg.REWARD_SUCCESS, True, "landed"

        progress = z - prev_z
        reward   = (self.cfg.REWARD_STEP_PENALTY
                    + self.cfg.REWARD_PROGRESS_COEF * progress)
        return reward, False, "alive"
