"""
envs/sb3_env_wrapper.py
-----------------------
Gymnasium-compatible wrapper around EnvClient for use with Stable Baselines3.

Translates our TCP-based env client into the gymnasium.Env interface that SB3
expects: reset() returns (obs, info), step() returns (obs, reward, terminated,
truncated, info).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.env_client import EnvClient


class MalmoGymEnv(gym.Env):
    """Wraps an EnvClient (TCP connection to env_server.py) as a gymnasium.Env."""

    metadata = {"render_modes": []}

    def __init__(self, cfg, port):
        super().__init__()
        self.cfg = cfg
        self.client = EnvClient(cfg.INPUT_SIZE, port=port)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.INPUT_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(cfg.N_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.client.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.client.step(int(action))
        # SB3/gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self):
        self.client.close()
