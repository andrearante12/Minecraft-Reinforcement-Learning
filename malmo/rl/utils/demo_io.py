"""
utils/demo_io.py
-----------------
Frame + action demonstration format for the vision world model.

A demo set is a DIRECTORY:

    <demo_dir>/index.json         {"format": "frames_v1", "env": ..., "frame_shape": [H,W,C],
                                   "episodes": [{"file", "steps", "outcome", "source"}, ...]}
    <demo_dir>/ep_0001.npz        one episode:
        jpeg_data     (total,) uint8   JPEG bytes of every frame, concatenated
        jpeg_offsets  (N+1,)   int64   frame t = jpeg_data[off[t]:off[t+1]]
        vecs          (N,D)    f32     proprio vector at step t
        actions       (N,)     i64     action taken FROM step t
        rewards       (N,)     f32     reward for that action
        dones         (N,)     f32
        privileged    (N,12)   f32     optional teacher-only target stream
    <demo_dir>/ep_0001.<encoder>_g<G>.npz   cached frozen-encoder tokens + thumbnails

Step t pairs the observation the action was chosen FROM with that action, the
same convention as SequenceReplayBuffer, so episodes load straight into it.
JPEG (~10 KB/frame at 224x224) keeps hundreds of demos on disk; raw frames are
kept (not just features) so swapping the encoder only means re-encoding.

Legacy vector-only JSON demos (utils/record_demos.py, generate_hunting_demos.py
without --vision) have no frames; `is_frame_demo_dir` lets callers tell them apart.
"""

import io
import json
import os

import numpy as np

FORMAT = "frames_v1"


def _jpeg_encode(frame, quality=90):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _jpeg_decode(data):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data.tobytes())).convert("RGB"), dtype=np.uint8)


def is_frame_demo_dir(path):
    return bool(path) and os.path.isfile(os.path.join(path, "index.json"))


def load_index(demo_dir):
    with open(os.path.join(demo_dir, "index.json")) as f:
        index = json.load(f)
    if index.get("format") != FORMAT:
        raise ValueError("{0}: unknown demo format {1!r}".format(demo_dir, index.get("format")))
    return index


class DemoWriter:
    """Append episodes to a demo directory (created if missing). Safe to reopen:
    new episodes continue the numbering and the index is rewritten after each."""

    def __init__(self, demo_dir, env_name, quality=90):
        self.dir = demo_dir
        self.quality = quality
        os.makedirs(demo_dir, exist_ok=True)
        if is_frame_demo_dir(demo_dir):
            self.index = load_index(demo_dir)
        else:
            self.index = {"format": FORMAT, "env": env_name, "frame_shape": None, "episodes": []}
        self._ep = None

    def __len__(self):
        return len(self.index["episodes"])

    def begin_episode(self):
        self._ep = {"jpeg": [], "vecs": [], "actions": [], "rewards": [], "dones": [], "priv": []}

    def add(self, frame, vec, action, reward, done, privileged=None):
        if self._ep is None:
            self.begin_episode()
        frame = np.asarray(frame, dtype=np.uint8)
        if self.index["frame_shape"] is None:
            self.index["frame_shape"] = list(frame.shape)
        self._ep["jpeg"].append(_jpeg_encode(frame, self.quality))
        self._ep["vecs"].append(np.asarray(vec, dtype=np.float32))
        self._ep["actions"].append(int(action))
        self._ep["rewards"].append(float(reward))
        self._ep["dones"].append(float(done))
        if privileged is not None:
            self._ep["priv"].append(np.asarray(privileged, dtype=np.float32))

    def end_episode(self, outcome, source="scripted"):
        """Write the buffered episode. Returns its file name, or None if empty."""
        ep, self._ep = self._ep, None
        if not ep or not ep["actions"]:
            return None
        n = len(ep["actions"])
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(j) for j in ep["jpeg"]])
        arrays = {
            "jpeg_data":    np.concatenate(ep["jpeg"]),
            "jpeg_offsets": offsets,
            "vecs":         np.stack(ep["vecs"]),
            "actions":      np.asarray(ep["actions"], dtype=np.int64),
            "rewards":      np.asarray(ep["rewards"], dtype=np.float32),
            "dones":        np.asarray(ep["dones"], dtype=np.float32),
        }
        if len(ep["priv"]) == n:
            arrays["privileged"] = np.stack(ep["priv"])
        name = "ep_{0:04d}.npz".format(len(self.index["episodes"]) + 1)
        np.savez(os.path.join(self.dir, name), **arrays)
        self.index["episodes"].append(
            {"file": name, "steps": n, "outcome": outcome, "source": source})
        tmp = os.path.join(self.dir, "index.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.index, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "index.json"))
        return name


def load_episode(demo_dir, entry, frames=True):
    """-> dict with vecs/actions/rewards/dones (+ frames (N,H,W,C) uint8, privileged)."""
    with np.load(os.path.join(demo_dir, entry["file"])) as z:
        ep = {k: z[k] for k in ("vecs", "actions", "rewards", "dones")}
        if "privileged" in z:
            ep["privileged"] = z["privileged"]
        if frames:
            data, off = z["jpeg_data"], z["jpeg_offsets"]
            ep["frames"] = np.stack([_jpeg_decode(data[off[t]:off[t + 1]])
                                     for t in range(len(off) - 1)])
    return ep


def iter_episodes(demo_dir, frames=True):
    index = load_index(demo_dir)
    for entry in index["episodes"]:
        yield entry, load_episode(demo_dir, entry, frames=frames)


def episode_features(demo_dir, entry, encoder, thumb_size=64):
    """(feats (N,T,D) fp16, thumbs (N,s,s,3) uint8) for one episode, cached on disk
    per encoder + pool grid so re-loading demos never re-runs the backbone."""
    from models.frozen_encoder import make_thumbs
    stem = os.path.splitext(entry["file"])[0]
    cache = os.path.join(demo_dir, "{0}.{1}_g{2}_t{3}.npz".format(
        stem, encoder.name, encoder.pool_grid, thumb_size))
    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["feats"], z["thumbs"]
    frames = load_episode(demo_dir, entry, frames=True)["frames"]
    feats = encoder.encode_np(frames)
    thumbs = make_thumbs(frames, thumb_size)
    np.savez(cache, feats=feats, thumbs=thumbs)
    return feats, thumbs
