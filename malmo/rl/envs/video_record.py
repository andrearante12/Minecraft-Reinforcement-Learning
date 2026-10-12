"""
Optional Malmo framebuffer capture for demo-site clips.

Off by default. When MALMO_RECORD_VIDEO is set in the env_server process,
requestVideo is turned on at MALMO_VIDEO_W x MALMO_VIDEO_H (default 640x360)
and frames are harvested next to each observation. Existing VIDEO_ENABLED
envs (hunting_video) keep their own size unless the env var overrides it.

MALMO_VIDEO_VIEWPOINT=1 renders the stream in third person (2 = facing).

Python 3.6-safe — imported by env_server.
"""

import os
import numpy as np


def apply_record_override(cfg):
    """Mutate cfg in place if MALMO_RECORD_VIDEO is set. Returns bool video-on."""
    if os.environ.get("MALMO_RECORD_VIDEO"):
        cfg.VIDEO_ENABLED = True
        cfg.VIDEO_WIDTH = int(os.environ.get("MALMO_VIDEO_W", "640"))
        cfg.VIDEO_HEIGHT = int(os.environ.get("MALMO_VIDEO_H", "360"))
        cfg.VIDEO_CHANNELS = int(getattr(cfg, "VIDEO_CHANNELS", 3) or 3)
    return bool(getattr(cfg, "VIDEO_ENABLED", False))


def request_video(mission, cfg):
    """mission.requestVideo + optional camera viewpoint.

    MALMO_VIDEO_VIEWPOINT: 0 = first person (default), 1 = third person behind,
    2 = facing the agent. Only the video stream changes (observations don't), but
    a pixel-input model (hunting_video) would see different frames — opt-in only.
    """
    mission.requestVideo(cfg.VIDEO_WIDTH, cfg.VIDEO_HEIGHT)
    vp = os.environ.get("MALMO_VIDEO_VIEWPOINT")
    if vp:
        mission.setViewpoint(int(vp))


def harvest_frame(env, ws):
    """Pull the latest Malmo video frame into env.last_frame, if any."""
    if not getattr(env, "_video", False):
        return
    if ws is None or len(ws.video_frames) == 0:
        return
    f = ws.video_frames[-1]
    ch = int(getattr(env.cfg, "VIDEO_CHANNELS", 3))
    env.last_frame = np.frombuffer(
        bytes(f.pixels), dtype=np.uint8
    ).reshape(f.height, f.width, ch).copy()
