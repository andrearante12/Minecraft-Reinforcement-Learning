"""
envs/bridging_env.py
--------------------
Gym-style Malmo wrapper for the bridging environment.
The agent must place blocks underneath itself to cross an open gap.

Key differences from ParkourEnv:
  - Persistent sneak state (toggle on/off across steps)
  - Inventory tracking (blocks remaining)
  - Ray-cast observation (what the crosshair targets)
  - Block placement detection via voxel grid diff
  - Bridge-specific reward shaping

Interface:
    env = BridgingEnv(cfg)
    obs = env.reset()                          # np.ndarray (INPUT_SIZE,)
    obs, reward, done, info = env.step(action)  # action is an int 0-13

Observation vector layout (INPUT_SIZE = 214):
    [0]      onGround
    [1]      yaw             (normalized -1 to 1)
    [2]      pitch           (normalized -1 to 1)
    [3]      delta_y
    [4]      delta_x
    [5]      delta_z
    [6]      inventory_count (normalized 0..1, count/64)
    [7]      ray_hit         (1 if looking at solid block face, else 0)
    [8]      ray_rel_x       (relative x of targeted block face)
    [9]      ray_rel_y       (relative y)
    [10]     ray_rel_z       (relative z)
    [11]     goal_dx
    [12]     goal_dy
    [13]     goal_dz
    [14:214] voxel grid      (5 x 5 x 8 = 200 values)
"""

import sys
import os
import math
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


class BridgingEnv:
    def __init__(self, cfg, malmo_port=None, force_reset=False):
        self.cfg       = cfg
        self.actions   = cfg.ACTIONS
        self.n_actions = cfg.N_ACTIONS
        self._malmo_port = malmo_port if malmo_port is not None else cfg.MALMO_PORT

        self._prev_pos = np.array(cfg.SPAWN, dtype=np.float32)
        self._goal_pos = np.array(cfg.GOAL_POS, dtype=np.float32)
        self._steps    = 0

        # Bridging-specific state
        self._prev_inv_count = 64          # previous inventory count for placement detection
        self._blocks_placed  = 0           # total blocks placed this episode
        self._max_z          = cfg.SPAWN[2]  # furthest Z the agent has reached
        self._landing_counter = 0
        self._sneaking       = False       # persistent sneak toggle state
        self._landing_active  = False

        # Shaping reward state
        self._steps_since_z_progress  = 0      # stall counter; reset whenever max_z increases
        self._entered_gap              = False  # first-gap-entry milestone fired this episode
        self._aligned_to_south_face    = False  # crosshair currently on a goal-facing bridge block face
        self._placed_while_aligned     = False  # block placed since current alignment began

        self.observation_shape = (cfg.INPUT_SIZE,)

        with open(cfg.MISSION_FILE, "r") as f:
            xml = f.read()
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
        self._steps           = 0
        self._prev_pos        = np.array(self.cfg.SPAWN, dtype=np.float32)
        self._prev_inv_count  = 64
        self._blocks_placed   = 0
        self._max_z           = self.cfg.SPAWN[2]
        self._landing_counter = 0
        self._landing_active  = False
        self._sneaking        = True   # start each episode already crouching

        self._steps_since_z_progress = 0
        self._entered_gap            = False
        self._aligned_to_south_face  = False
        self._placed_while_aligned   = False
        try:
            self._agent_host.sendCommand("crouch 0")  # release any lingering crouch first
        except Exception:
            pass

        time.sleep(0.5)
        if self._next_force_reset:
            self._start_mission(self._mission_xml_force)
            self._next_force_reset = False
        else:
            self._start_mission(self._mission_xml_fast)

        try:
            self._agent_host.sendCommand("crouch 1")  # begin episode crouching
        except Exception:
            pass

        return self._get_observation()

    def step(self, action):
        self._steps += 1
        prev_pos = self._prev_pos.copy()

        self._take_action(action)

        obs_dict, world_state = self._get_obs_dict()
        obs = self._build_obs_vector(obs_dict)  # updates self._prev_pos

        # Debug: print hotbar/inventory keys on first step to verify key name
        if self._steps == 1:
            hotbar_keys = [k for k in obs_dict if "otbar" in k or "nventory" in k or "lot" in k.lower()]
            print("DEBUG bridging obs keys (hotbar/inv): {0}".format(hotbar_keys))

        # Detect block placement via inventory count drop.
        # Placement validity is judged by agent z-coordinate (Malmo obs lag
        # makes the placed-block voxel unreliable on the same step).
        inv_count   = int(obs_dict.get("Hotbar_0_size", self._prev_inv_count))
        blocks_used = 0
        placement_reward = 0.0
        if inv_count < self._prev_inv_count:
            blocks_used = self._prev_inv_count - inv_count
            self._blocks_placed += blocks_used
            z_now = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))
            y_now = float(obs_dict.get("YPos", self.cfg.SPAWN[1]))
            in_gap_range = ((self.cfg.BRIDGE_Z_START - 1.0)
                            <= z_now
                            <= (self.cfg.BRIDGE_Z_END + 1.0))
            at_bridge_level = abs(y_now - (self.cfg.BRIDGE_Y + 1.0)) < 0.5
            if in_gap_range and at_bridge_level:
                placement_reward = self.cfg.REWARD_BLOCK_PLACED_VALID * blocks_used
                if self._sneaking:
                    placement_reward += self.cfg.REWARD_SNEAK_PLACE * blocks_used
            else:
                placement_reward = self.cfg.REWARD_BLOCK_PLACED_WASTED * blocks_used
        self._prev_inv_count = inv_count

        reward, done, outcome = self._get_reward(obs_dict, prev_pos)

        # Add placement and behavioural shaping rewards (only while alive)
        if not done:
            reward += placement_reward
            reward += self._get_shaping_reward(obs_dict, blocks_used)

        # Mission ended unexpectedly
        if not world_state.is_mission_running and not done:
            done, outcome = True, "mission_ended"
            if self.cfg.REWARD_ON_MISSION_ENDED:
                reward = self.cfg.REWARD_TIMEOUT
                if self.cfg.PROXIMITY_SCALED_TERMINAL:
                    pos_now = self._current_pos(obs_dict)
                    reward *= (1.0 - self._proximity(pos_now))

        # Timeout
        if self._steps >= self.cfg.MAX_STEPS and not done:
            z_now = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))
            if z_now >= (self.cfg.Z_SUCCESS - self.cfg.NEAR_MISS_THRESHOLD):
                reward = self.cfg.REWARD_NEAR_MISS
                done, outcome = True, "near_miss"
            else:
                done, outcome = True, "timeout"
                reward = self.cfg.REWARD_TIMEOUT
                if self.cfg.PROXIMITY_SCALED_TERMINAL:
                    pos_now = self._current_pos(obs_dict)
                    reward *= (1.0 - self._proximity(pos_now))

        if done:
            try:
                self._agent_host.sendCommand("quit")
            except Exception:
                pass

        info = {
            "outcome":       outcome,
            "steps":         self._steps,
            "pos":           (obs_dict.get("XPos", 0),
                              obs_dict.get("YPos", 0),
                              obs_dict.get("ZPos", 0)),
            "yaw":           float(obs_dict.get("Yaw",      0.0)),
            "pitch":         float(obs_dict.get("Pitch",    0.0)),
            "on_ground":     int(obs_dict.get("OnGround",   False)),
            "action":        self.actions[action][0],
            "blocks_placed": self._blocks_placed,
        }
        return obs, reward, done, info

    def close(self):
        try:
            self._agent_host.sendCommand("quit")
        except Exception:
            pass

    # ── Action execution ──────────────────────────────────────────────────────

    def _take_action(self, action_idx):
        name, cmds_on, cmds_off = self.actions[action_idx]
        # Cancel any residual camera velocity from Minecraft's native mouse look
        # which is active during the gap between env steps.
        self._agent_host.sendCommand("pitch 0")
        self._agent_host.sendCommand("turn 0")

        # Update persistent sneak state for explicit sneak actions.
        if name == "sneak_down":
            self._sneaking = True
        elif name == "sneak_up":
            self._sneaking = False

        for cmd in cmds_on:
            self._agent_host.sendCommand(cmd)
        time.sleep(self.cfg.STEP_DURATION)
        for cmd in cmds_off:
            self._agent_host.sendCommand(cmd)

        # Re-apply persistent sneak after movement/placement actions that may
        # implicitly reset crouch state in some Malmo builds. Skip for sneak
        # actions themselves — they already set crouch explicitly via cmds_on.
        if self._sneaking and name not in ("sneak_down", "sneak_up"):
            self._agent_host.sendCommand("crouch 1")

    # ── Malmo interaction (copied from ParkourEnv) ────────────────────────────

    def _start_mission(self, mission_xml=None, max_retries=3,
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

    # ── Observation building ──────────────────────────────────────────────────

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

        # Base proprioception (9): onGround, yaw, pitch, velocity, absolute position
        norm_x = x - self.cfg.SPAWN[0]
        norm_y = y - self.cfg.SPAWN[1]
        norm_z = z - self.cfg.SPAWN[2]
        base_proprio = [gnd, yaw / 180.0, pit / 90.0, vel[1], vel[0], vel[2],
                        norm_x, norm_y, norm_z]

        # Bridging-specific observations (5)
        inv_count = float(obs.get("Hotbar_0_size", 0)) / 64.0

        # Ray-cast: what the crosshair is pointing at (7 values)
        los = obs.get("LineOfSight", {})
        if los and los.get("hitType") == "block":
            ray_hit   = 1.0
            ray_rel_x = float(los.get("x", 0)) - x
            ray_rel_y = float(los.get("y", 0)) - y
            ray_rel_z = float(los.get("z", 0)) - z
            # Face encoding: vertical axis (+1 top, -1 bottom, 0 side),
            # horizontal dx (+1 east, -1 west, 0), horizontal dz (+1 south, -1 north, 0)
            face = los.get("face", "")
            face_v  = +1.0 if face == "top"   else (-1.0 if face == "bottom" else 0.0)
            face_dx = +1.0 if face == "east"  else (-1.0 if face == "west"   else 0.0)
            face_dz = +1.0 if face == "south" else (-1.0 if face == "north"  else 0.0)
        else:
            ray_hit   = 0.0
            ray_rel_x = 0.0
            ray_rel_y = 0.0
            ray_rel_z = 0.0
            face_v    = 0.0
            face_dx   = 0.0
            face_dz   = 0.0

        extra_proprio = [inv_count, ray_hit,
                         ray_rel_x, ray_rel_y, ray_rel_z,
                         face_v, face_dx, face_dz]

        proprio = np.array(base_proprio + extra_proprio, dtype=np.float32)

        # Goal delta (3)
        goal_delta = (self._goal_pos - pos).astype(np.float32)

        # Voxel grid
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

    # ── Reward function ───────────────────────────────────────────────────────

    def _get_shaping_reward(self, obs_dict, blocks_placed_this_step):
        """Behavioural shaping rewards that sit on top of the base step/placement/progress signals.

        Called from step() only when the episode is still alive (not done).

        Rewards computed here:
          - First-gap-entry milestone (one-time)
          - Sneak-in-gap / sneak-at-edge (per step)
          - Stall penalty (per step, once stall counter exceeds threshold)
          - Crosshair alignment break (one-time per look-away-without-placing event)

        State managed across steps (initialised/reset in __init__ and reset()):
          self._entered_gap            -- milestone has fired this episode
          self._aligned_to_south_face  -- crosshair currently on a goal-facing bridge face
          self._placed_while_aligned   -- block placed since the current alignment began
          self._steps_since_z_progress -- stall counter, updated in _get_reward()
        """
        reward = 0.0
        z = float(obs_dict.get("ZPos", self.cfg.SPAWN[2]))

        in_gap  = self.cfg.BRIDGE_Z_START <= z <= self.cfg.BRIDGE_Z_END + 1.0
        at_edge = (self.cfg.BRIDGE_Z_START - 1.0) <= z < self.cfg.BRIDGE_Z_START

        # ── First-gap-entry milestone ──────────────────────────────────────
        if in_gap and not self._entered_gap:
            self._entered_gap = True
            reward += self.cfg.REWARD_ENTERED_GAP

        # ── Sneak bonuses ──────────────────────────────────────────────────
        # Only reward sneak when the agent is at the +Z (goal-facing) side of
        # their current block — i.e. at the leading edge of the bridge.
        # For gap blocks:         z >= self._max_z - 0.5  (front half of furthest block)
        # For the spawn platform: z >= BRIDGE_Z_START - 0.5 (front half of last solid block)
        if self._sneaking:
            at_leading_edge = (
                (in_gap  and z >= self._max_z - 0.5) or
                (at_edge and z >= self.cfg.BRIDGE_Z_START - 0.5)
            )
            if at_leading_edge:
                if in_gap:
                    reward += self.cfg.REWARD_SNEAK_IN_GAP
                elif at_edge:
                    reward += self.cfg.REWARD_SNEAK_AT_EDGE

        # ── Camera shaping (look-down + look-back) ─────────────────────────
        # Encourages the crosshair to land on the south face of the last bridge
        # block, where the next block should be placed.
        # Only active while in placement-relevant zone (gap or just before it).
        if in_gap or at_edge:
            pitch = float(obs_dict.get("Pitch", 0.0))  # 0=horizontal, 90=straight down
            yaw   = float(obs_dict.get("Yaw",   0.0))  # 0=south(+Z), 180=north(-Z/back)

            # Look-down: sin(π·pitch/90) peaks at 45°, falls to 0 at both 0°
            # and 90°, so the policy is pushed toward an intermediate angle.
            if 0.0 < pitch < 90.0:
                reward += self.cfg.REWARD_LOOK_DOWN * math.sin(math.pi * pitch / 90.0)

            # Look-back: (-cos(yaw)+1)/2 is 0 when facing south (+Z/goal),
            # 1 when facing north (-Z/backward).  Handles ±180 wraparound cleanly.
            look_back_factor = (-math.cos(math.radians(yaw)) + 1.0) / 2.0
            reward += self.cfg.REWARD_LOOK_BACK * look_back_factor

        # ── Stall penalty ──────────────────────────────────────────────────
        if self._steps_since_z_progress > self.cfg.STALL_THRESHOLD:
            reward += self.cfg.REWARD_STALL

        # ── Crosshair alignment ────────────────────────────────────────────
        # "Correct face" = the face of a bridge-zone block that points toward
        # the goal (+Z direction = Malmo face name "south").  Once the agent
        # begins looking at such a face, looking away without first placing a
        # block incurs REWARD_ALIGNMENT_BREAK.  A successful placement resets
        # the state cleanly so the immediately following look-away (caused by
        # the block appearing) is never penalised.
        los = obs_dict.get("LineOfSight", {})
        curr_aligned = False
        if los and los.get("hitType") == "block" and los.get("face") == "south":
            target_x = float(los.get("x", 0))
            target_y = float(los.get("y", 0))
            target_z = float(los.get("z", 0))
            if (self.cfg.BRIDGE_Z_START - 1 <= target_z <= self.cfg.BRIDGE_Z_END and
                    abs(target_y - self.cfg.BRIDGE_Y) < 1.5 and
                    abs(target_x) < 1.5):
                curr_aligned = True

        prev_aligned = self._aligned_to_south_face

        # A block placed while previously aligned protects the look-away that
        # follows (the block appearing causes the crosshair to shift off the face).
        if blocks_placed_this_step > 0 and prev_aligned:
            self._placed_while_aligned = True

        # Look-away transition
        if prev_aligned and not curr_aligned:
            if not self._placed_while_aligned:
                reward += self.cfg.REWARD_ALIGNMENT_BREAK   # negative constant
            self._placed_while_aligned = False

        # New alignment: reset placed flag so a fresh look-then-place cycle begins
        if not prev_aligned and curr_aligned:
            self._placed_while_aligned = False

        self._aligned_to_south_face = curr_aligned

        return reward

    def _current_pos(self, obs_dict):
        return np.array([
            float(obs_dict.get("XPos", self.cfg.SPAWN[0])),
            float(obs_dict.get("YPos", self.cfg.SPAWN[1])),
            float(obs_dict.get("ZPos", self.cfg.SPAWN[2])),
        ], dtype=np.float32)

    def _proximity(self, pos):
        """Return 0.0 (at spawn) to 1.0 (at goal) based on distance to goal."""
        initial_dist = np.linalg.norm(
            np.array(self.cfg.SPAWN, dtype=np.float32)
            - np.array(self.cfg.GOAL_POS, dtype=np.float32)
        )
        current_dist = np.linalg.norm(pos - np.array(self.cfg.GOAL_POS, dtype=np.float32))
        if initial_dist == 0:
            return 1.0
        return max(1.0 - current_dist / initial_dist, 0.0)

    def _in_landing_zone(self, x, y, z, on_ground=False):
        if z < self.cfg.Z_SUCCESS:
            return False
        if self.cfg.Z_SUCCESS_MAX is not None and z > self.cfg.Z_SUCCESS_MAX:
            return False
        if y < self.cfg.FALL_Y_THRESHOLD:
            return False
        if x < self.cfg.BRIDGE_X_MIN - 1.0 or x > self.cfg.BRIDGE_X_MAX + 1.0:
            return False
        return True

    def _get_reward(self, obs, prev_pos):
        x = float(obs.get("XPos", self.cfg.SPAWN[0]))
        y = float(obs.get("YPos", self.cfg.SPAWN[1]))
        z = float(obs.get("ZPos", self.cfg.SPAWN[2]))

        # Fell into the void
        if y < self.cfg.FALL_Y_THRESHOLD:
            reward = self.cfg.REWARD_FELL
            if self.cfg.PROXIMITY_SCALED_TERMINAL:
                reward *= (1.0 - self._proximity(np.array([x, y, z])))
            self._landing_active = False
            self._landing_counter = 0
            return reward, True, "fell"

        on_ground = bool(obs.get("OnGround", False))

        # Landing phase (must stay on end platform)
        if self._landing_active:
            if self._in_landing_zone(x, y, z, on_ground):
                self._landing_counter += 1
                if self._landing_counter >= self.cfg.LANDING_TICKS:
                    return self.cfg.REWARD_SUCCESS, True, "landed"
                return self.cfg.REWARD_LANDING_TICK, False, "alive"
            else:
                self._landing_active = False
                self._landing_counter = 0
                return self.cfg.REWARD_FELL, True, "fell"

        # Success check — reached the end platform
        if self._in_landing_zone(x, y, z, on_ground):
            if self.cfg.LANDING_TICKS > 0:
                self._landing_active = True
                self._landing_counter = 0
                return 0.0, False, "alive"
            else:
                return self.cfg.REWARD_SUCCESS, True, "landed"

        # Z-progress reward: bonus for advancing to new Z positions.
        # Also drives the stall counter used by _get_shaping_reward.
        z_progress_reward = 0.0
        if z > self._max_z:
            # Cap reward at Z_SUCCESS — no incentive to bridge past the goal
            z_capped = min(z, self.cfg.Z_SUCCESS)
            if z_capped > self._max_z:
                z_progress_reward = self.cfg.REWARD_PROGRESS_COEF * (z_capped - self._max_z)
            self._max_z = z
            self._steps_since_z_progress = 0
        else:
            self._steps_since_z_progress += 1

        reward = self.cfg.REWARD_STEP_PENALTY + z_progress_reward
        return reward, False, "alive"
