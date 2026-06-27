"""
algos/replay_buffer.py
----------------------
A simple FIFO replay buffer of (obs, action, reward, next_obs, done) tuples.

This was originally defined inside dqn.py; it is factored out here because the
world-model agent (dreamer.py) trains on the same full transitions. DQN imports
ReplayBuffer from this module, so its behaviour is unchanged.

Extra method for model-based RL:
  - sample_starts(): returns a batch of observations only, used as the root
    states from which the world model rolls out imagined trajectories.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def sample_starts(self, batch_size):
        """Return a batch of start observations (for imagination rollouts)."""
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        return np.array([t[0] for t in batch], dtype=np.float32)

    def __len__(self):
        return len(self.buffer)
