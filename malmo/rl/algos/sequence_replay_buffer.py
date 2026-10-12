"""
algos/sequence_replay_buffer.py
--------------------------------
Episode-aware sequence buffer for the video world model (used only by
algos/dreamer_video.py). A SEPARATE class from ReplayBuffer (replay_buffer.py)
— it is not touched, and its existing consumers (dqn.py, dreamer.py) are
unaffected.

Differences from ReplayBuffer that matter for images:
  - Frames are stored ONCE per step (no next_obs duplication) as uint8 — NOT
    cast to float32 (that cast, fine for a 98-float vector, would 4x the
    memory of an image and is unnecessary since the model normalizes on GPU).
  - Storage is per-episode so training can sample whole (frame, vec, action,
    reward, done) windows with proper masking, which single-transition
    ReplayBuffer cannot do.
  - Short episodes are NOT discarded — a fast kill is exactly the data the
    hunting world model most needs, so windows shorter than seq_len are
    zero-padded and masked rather than dropped.

The per-step observation is a dict of arrays whose dtype is preserved:
`add()` stores {"frames": uint8 frame} (pixel world model); `add_obs()` stores
any keys, e.g. {"feats": fp16 frozen-encoder tokens, "thumbs": uint8 64x64}
for the feature world model. `sample_sequences` returns every stored key.
"""

import numpy as np
from collections import deque


class SequenceReplayBuffer:
    def __init__(self, capacity, seq_len, min_seq=8):
        """capacity: max total STEPS kept (whole episodes evicted oldest-first
        once exceeded). seq_len: training window length. min_seq: shortest
        episode usable as a sample source (shorter ones are still stored, just
        not sampled, since even a padded window from them would be mostly
        padding)."""
        self.capacity = capacity
        self.seq_len  = seq_len
        self.min_seq  = min_seq
        self.episodes = deque()
        self._current = None
        self._total_steps = 0

    _STEP_KEYS = ("vecs", "actions", "rewards", "dones")

    def add(self, frame, vec, action, reward, done):
        self.add_obs({"frames": np.asarray(frame, dtype=np.uint8)}, vec, action, reward, done)

    def add_obs(self, obs, vec, action, reward, done):
        """obs: dict of per-step arrays (dtype kept as given)."""
        if self._current is None:
            self._current = {k: [] for k in self._STEP_KEYS}
            self._current.update({k: [] for k in obs})
        for k, v in obs.items():
            self._current[k].append(np.asarray(v))
        self._current["vecs"].append(np.asarray(vec, dtype=np.float32))
        self._current["actions"].append(int(action))
        self._current["rewards"].append(float(reward))
        self._current["dones"].append(float(done))
        self._total_steps += 1
        if done:
            self._seal_current()
        self._evict()

    def add_episode(self, obs, vecs, actions, rewards, dones):
        """Append a whole finished episode (demos). obs: dict of (N,...) arrays."""
        n = len(actions)
        if n == 0:
            return
        self._seal_current()
        ep = {k: list(np.asarray(v)) for k, v in obs.items()}
        ep["vecs"]    = list(np.asarray(vecs, dtype=np.float32))
        ep["actions"] = [int(a) for a in actions]
        ep["rewards"] = [float(r) for r in rewards]
        ep["dones"]   = [float(d) for d in dones]
        self.episodes.append(ep)
        self._total_steps += n
        self._evict()

    def obs_keys(self):
        src = self.episodes[0] if self.episodes else self._current
        return [] if src is None else [k for k in src if k not in self._STEP_KEYS]

    def has_samples(self):
        return any(len(ep["actions"]) >= self.min_seq for ep in self.episodes)

    def _seal_current(self):
        if self._current is not None and len(self._current["actions"]) > 0:
            self.episodes.append(self._current)
        self._current = None

    def _evict(self):
        # Keep at least one sealed episode even if it alone exceeds capacity
        # (a single long episode must never empty the buffer).
        while self._total_steps > self.capacity and len(self.episodes) > 1:
            oldest = self.episodes.popleft()
            self._total_steps -= len(oldest["actions"])

    def __len__(self):
        return self._total_steps

    def sample_sequences(self, batch_size):
        """Sample `batch_size` windows of length `seq_len` from sealed episodes,
        weighted by episode length. Windows shorter than seq_len (episode ends
        before the window fills) are zero-padded on the right; `mask` marks
        real (1) vs padded (0) steps.

        Returns a dict of numpy arrays (one entry per stored obs key):
          frames  (B,L,H,W,C) uint8   vecs (B,L,D) f32   actions (B,L) i64
          rewards (B,L) f32   dones (B,L) f32   mask (B,L) f32
        """
        usable = [ep for ep in self.episodes if len(ep["actions"]) >= self.min_seq]
        if not usable:
            raise ValueError(
                "SequenceReplayBuffer: no sealed episode reaches min_seq={0} yet "
                "({1} sealed episodes so far)".format(self.min_seq, len(self.episodes)))

        lengths = np.array([len(ep["actions"]) for ep in usable], dtype=np.float64)
        probs = lengths / lengths.sum()

        obs_keys = [k for k in usable[0] if k not in self._STEP_KEYS]
        vec_dim  = usable[0]["vecs"][0].shape[0]

        obs = {k: np.zeros((batch_size, self.seq_len) + usable[0][k][0].shape,
                           dtype=usable[0][k][0].dtype) for k in obs_keys}
        vecs    = np.zeros((batch_size, self.seq_len, vec_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rewards = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        dones   = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        mask    = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        ep_idxs = np.random.choice(len(usable), size=batch_size, p=probs)
        for i, ep_idx in enumerate(ep_idxs):
            ep = usable[ep_idx]
            L = len(ep["actions"])
            max_start = max(0, L - self.seq_len)
            start = np.random.randint(0, max_start + 1)
            n = min(self.seq_len, L - start)
            for t in range(n):
                for k in obs_keys:
                    obs[k][i, t] = ep[k][start + t]
                vecs[i, t]    = ep["vecs"][start + t]
                actions[i, t] = ep["actions"][start + t]
                rewards[i, t] = ep["rewards"][start + t]
                dones[i, t]   = ep["dones"][start + t]
                mask[i, t]    = 1.0

        batch = {"vecs": vecs, "actions": actions,
                 "rewards": rewards, "dones": dones, "mask": mask}
        batch.update(obs)
        return batch
