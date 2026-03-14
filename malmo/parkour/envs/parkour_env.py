"""
envs/parkour_env.py
-------------------
Gym-style wrapper around Malmo for the parkour task.

PPO and all other RL code interact ONLY with this interface — MalmoPython
is never imported anywhere else. This separation means:
  - You can test PPO on CartPole without Malmo installed
  - You can swap maps by changing config without touching the algorithm
  - The observation building, reward shaping, and action execution
    are all in one place and easy to modify

Interface:
    env = ParkourEnv(cfg)
    obs = env.reset()                         # np.ndarray (INPUT_SIZE,)
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

# ── Malmo import ──────────────────────────────────────────────────────────────
from training.config import CFG
sys.path.insert(0, os.path.abspath(CFG.MALMO_PYTHON))

import MalmoPython


class ParkourEnv:
    """
    Gym-style Malmo environment for the parkour jump task.
    """

    def __init__(self, cfg=CFG):
        self.cfg       = cfg
        self.actions   = cfg.ACTIONS
        self.n_actions = cfg.N_ACTIONS

        # Track previous position for velocity computation
        self._prev_pos = np.array(cfg.SPAWN, dtype=np.float32)
        self._goal_pos = np.array(cfg.GOAL_POS, dtype=np.float32)

        # Track previous Z separately for clean progress reward computation
        self._prev_z = float(cfg.SPAWN[2])

        # Step counter for the current episode
        self._steps = 0

        # Observation space shape — useful for building the network
        self.observation_shape = (cfg.INPUT_SIZE,)

        # ── Load and patch mission XML ─────────────────────────────────────
        with open(cfg.MISSION_FILE, "r") as f:
            xml = f.read()

        # Creative mode: no death
        # forceReset=false: reuse world between episodes for speed
        # xml = xml.replace('mode="Survival"',   'mode="Creative"')
        xml = xml.replace('forceReset="true"', 'forceReset="false"')
        self._mission_xml = xml

        # ── Initialize Malmo agent host ────────────────────────────────────
        self._agent_host = MalmoPython.AgentHost()
        try:
            self._agent_host.parse(sys.argv)
        except RuntimeError as e:
            print("ERROR parsing AgentHost:", e)
            sys.exit(1)

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        Restarts the Malmo mission every episode for a guaranteed clean
        respawn at the XML <Placement> position.

        Returns:
            obs: np.ndarray of shape (INPUT_SIZE,)
        """
        self._steps  = 0
        self._prev_pos = np.array(self.cfg.SPAWN, dtype=np.float32)
        self._prev_z   = float(self.cfg.SPAWN[2])

        # Small gap so Malmo doesn't get confused between episodes
        time.sleep(0.5)
        self._start_mission()

        obs = self._get_observation()
        return obs

    def step(self, action: int):
        """
        Execute one action in the environment.

        Args:
            action: int index into CFG.ACTIONS

        Returns:
            obs:    np.ndarray (INPUT_SIZE,)
            reward: float
            done:   bool
            info:   dict with debug info
        """
        self._steps += 1

        # Save Z before action for progress reward
        prev_z = self._prev_z

        # Execute the action
        self._take_action(action)

        # Get new observation
        obs_dict, world_state = self._get_obs_dict()

        # Update prev_z for next step
        self._prev_z = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))

        # Build observation vector
        obs = self._build_obs_vector(obs_dict)

        # Compute reward and termination
        reward, done, outcome = self._get_reward(obs_dict, prev_z)

        # Terminate if mission ended externally (time limit etc.)
        if not world_state.is_mission_running:
            done    = True
            outcome = "mission_ended"

        # Max steps timeout
        if self._steps >= self.cfg.MAX_STEPS:
            done    = True
            outcome = "timeout"

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
        """Send quit to cleanly end the mission."""
        try:
            self._agent_host.sendCommand("quit")
        except Exception:
            pass

    # ── Malmo interaction ──────────────────────────────────────────────────────

    def _start_mission(self, max_retries: int = 3):
        # ── Wait for any existing mission to end first ─────────────────────
        print("Waiting for previous mission to end...", end=" ")
        ws = self._agent_host.getWorldState()
        while ws.is_mission_running:
            print(".", end="", flush=True)
            time.sleep(0.1)
            ws = self._agent_host.getWorldState()
        print(" done.")
        time.sleep(0.5)  # extra buffer for Malmo to fully release the client

        mission        = MalmoPython.MissionSpec(self._mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        for attempt in range(max_retries):
            try:
                self._agent_host.startMission(mission, mission_record)
                break
            except RuntimeError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Could not start mission after {0} attempts: {1}\n"
                        "Is Minecraft running on port {2}?".format(
                            max_retries, e, self.cfg.MALMO_PORT))
                print("  Retrying mission start ({0}/{1})...".format(
                    attempt + 1, max_retries))
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

    def _take_action(self, action_idx: int):
        """Send action commands, hold for STEP_DURATION, then release."""
        _, cmds_on, cmds_off = self.actions[action_idx]
        for cmd in cmds_on:
            self._agent_host.sendCommand(cmd)
        time.sleep(self.cfg.STEP_DURATION)
        for cmd in cmds_off:
            self._agent_host.sendCommand(cmd)

    def _get_obs_dict(self, timeout: float = 3.0):
        """
        Poll Malmo until a fresh observation arrives.
        Returns (obs_dict, world_state). obs_dict is empty if timeout.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            ws = self._agent_host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                return json.loads(ws.observations[-1].text), ws
            time.sleep(0.03)
        return {}, self._agent_host.getWorldState()

    def _get_observation(self) -> np.ndarray:
        """Convenience wrapper — get obs dict and build vector."""
        obs_dict, _ = self._get_obs_dict()
        return self._build_obs_vector(obs_dict)

    # ── Observation building ───────────────────────────────────────────────────

    def _build_obs_vector(self, obs: dict) -> np.ndarray:
        """
        Convert Malmo's JSON observation dict into a flat float32 numpy array.

        Layout:
            [0:6]    proprioception
            [6:9]    goal delta
            [9:129]  voxel grid
        """
        x   = float(obs.get("XPos",     self.cfg.SPAWN[0]))
        y   = float(obs.get("YPos",     self.cfg.SPAWN[1]))
        z   = float(obs.get("ZPos",     self.cfg.SPAWN[2]))
        yaw = float(obs.get("Yaw",      0.0))
        pit = float(obs.get("Pitch",    0.0))
        gnd = float(obs.get("OnGround", False))

        pos            = np.array([x, y, z], dtype=np.float32)
        prev_pos       = self._prev_pos.copy()   # save before updating
        vel            = pos - prev_pos
        self._prev_pos = pos

        # Proprioception (6)
        proprio = np.array([
            gnd,
            yaw / 180.0,   # normalize to [-1, 1]
            pit / 90.0,    # normalize to [-1, 1]
            vel[1],        # dy — vertical velocity
            vel[0],        # dx
            vel[2],        # dz — forward velocity, most important
        ], dtype=np.float32)

        # Goal delta (3)
        goal_delta = (self._goal_pos - pos).astype(np.float32)

        # Voxel grid (120)
        raw_grid = obs.get("floor3x3", [])
        voxels   = self._encode_grid(raw_grid)

        return np.concatenate([proprio, goal_delta, voxels])

    def _encode_grid(self, raw_grid: list) -> np.ndarray:
        """
        Convert list of block name strings to encoded float array.
        air=0, solid=1, goal_block=2
        """
        expected = self.cfg.GRID_SIZE
        if len(raw_grid) != expected:
            # Return all-air if grid is missing or wrong size
            return np.zeros(expected, dtype=np.float32)

        encoded = np.zeros(expected, dtype=np.float32)
        for i, block in enumerate(raw_grid):
            encoded[i] = float(self.cfg.BLOCK_ENCODING.get(block, 1))
        return encoded

    # ── Reward function ────────────────────────────────────────────────────────

    def _get_reward(self, obs: dict, prev_z: float):
        """
        Compute reward and termination signal.

        Args:
            obs:    Malmo observation dict
            prev_z: agent Z position before this step (for progress reward)

        Returns:
            reward:  float
            done:    bool
            outcome: str ("fell", "landed", "alive")
        """
        y = float(obs.get("YPos", self.cfg.SPAWN[1]))
        z = float(obs.get("ZPos", self.cfg.SPAWN[2]))

        # Fell off the platform
        if y < self.cfg.FALL_Y_THRESHOLD:
            return self.cfg.REWARD_FELL, True, "fell"

        # Crossed the gap successfully
        if z >= self.cfg.Z_SUCCESS:
            return self.cfg.REWARD_SUCCESS, True, "landed"

        # Dense progress reward — reward forward movement toward goal
        # progress is positive when Z increases (moving toward gap)
        progress = z - prev_z
        reward   = (self.cfg.REWARD_STEP_PENALTY
                    + self.cfg.REWARD_PROGRESS_COEF * progress)

        return reward, False, "alive"