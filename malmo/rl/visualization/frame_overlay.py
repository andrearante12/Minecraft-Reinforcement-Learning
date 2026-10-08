"""
visualization/frame_overlay.py
------------------------------
Draw the world model's IMAGINED future directly onto a first-person frame.

Pinhole camera model of Minecraft's player view: the camera sits at the eye
(feet + 1.62), looks along (yaw, pitch), and has a vertical field of view
`fov` (Minecraft default 70). World points (x, y, z) are projected into pixel
coordinates and drawn as ghost markers / trails over the frame.

Camera pose comes from the observation itself (pos-rel-spawn, yaw, pitch), so
the live entry point only needs the Malmo frame + obs:

    img = overlay_from_obs(frame_rgb, real_obs, imag_obs_seq, cfg,
                           members_obs=[...], step_label="t+3")

Conventions (Malmo): yaw 0 faces +Z, positive yaw turns toward -X;
pitch > 0 looks down. forward = (-sin(yaw)cos(p), -sin(p), cos(yaw)cos(p)).

NOT yet validated against real Malmo frames — the camera constants (FOV, eye
height) and sync between frame and obs need a live check. `render_synthetic_frame`
draws a stand-in first-person scene so the overlay can be exercised offline.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

EYE_HEIGHT = 1.62
DEFAULT_FOV = 70.0       # vertical, degrees
NEAR = 0.1


# ── camera ───────────────────────────────────────────────────────────────────
def camera_basis(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
    r = np.array([-math.cos(y), 0.0, -math.sin(y)])
    u = np.cross(r, f)
    return f, r, u


def to_camera(points, cam_pos, yaw_deg, pitch_deg):
    """World points (N,3) → camera coords (N,3) as (right, up, forward)."""
    f, r, u = camera_basis(yaw_deg, pitch_deg)
    d = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(cam_pos, dtype=np.float64)
    return np.stack([d @ r, d @ u, d @ f], axis=1)


def to_pixels(cam_pts, W, H, fov=DEFAULT_FOV):
    """Camera coords → (px, py) (NaN where behind the near plane)."""
    fpx = (H / 2.0) / math.tan(math.radians(fov) / 2.0)
    z = cam_pts[:, 2]
    ok = z > NEAR
    zz = np.where(ok, z, 1.0)
    px = W / 2.0 + fpx * cam_pts[:, 0] / zz
    py = H / 2.0 - fpx * cam_pts[:, 1] / zz
    px[~ok] = np.nan; py[~ok] = np.nan
    return np.stack([px, py], axis=1)


def project(points, cam_pos, yaw_deg, pitch_deg, W, H, fov=DEFAULT_FOV):
    return to_pixels(to_camera(points, cam_pos, yaw_deg, pitch_deg), W, H, fov)


def _clip_segment(a, b):
    """Clip camera-space segment to z >= NEAR. Returns (a, b) or None."""
    za, zb = a[2], b[2]
    if za < NEAR and zb < NEAR:
        return None
    if za < NEAR:
        a = a + (b - a) * ((NEAR - za) / (zb - za))
    elif zb < NEAR:
        b = b + (a - b) * ((NEAR - zb) / (za - zb))
    return a, b



def _clip_rect(p0, p1, W, H, m=2.0):
    """Liang–Barsky clip of a 2D segment to the image rect (+margin). None if outside."""
    x0, y0 = float(p0[0]), float(p0[1]); x1, y1 = float(p1[0]), float(p1[1])
    if not all(map(math.isfinite, (x0, y0, x1, y1))):
        return None
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 + m), (dx, W + m - x0), (-dy, y0 + m), (dy, H + m - y0)):
        if pp == 0:
            if qq < 0:
                return None
        else:
            t = qq / pp
            if pp < 0:
                if t > t1: return None
                t0 = max(t0, t)
            else:
                if t < t0: return None
                t1 = min(t1, t)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def safe_line(draw, p0, p1, W, H, **kw):
    seg = _clip_rect(p0, p1, W, H)
    if seg is not None:
        draw.line([seg[0], seg[1]], **kw)

# ── drawing primitives ───────────────────────────────────────────────────────
def draw_polyline(draw, pts_world, cam, W, H, fov, color, width=2, dash=None):
    cam_pts = to_camera(pts_world, cam["pos"], cam["yaw"], cam["pitch"])
    n = 0
    for i in range(len(cam_pts) - 1):
        seg = _clip_segment(cam_pts[i], cam_pts[i + 1])
        if seg is None:
            continue
        px = to_pixels(np.stack(seg), W, H, fov)
        n += 1
        if dash and n % dash == 0:
            continue
        safe_line(draw, px[0], px[1], W, H, fill=color, width=width)


def draw_box(draw, center, cam, W, H, fov, color, size=(0.9, 0.9, 0.9), fill=None, width=2):
    """Axis-aligned box centred (x,z) on the ground (y = feet). Wire or filled faces."""
    cx, cy, cz = center
    sx, sy, sz = size
    xs, ys, zs = [cx - sx / 2, cx + sx / 2], [cy, cy + sy], [cz - sz / 2, cz + sz / 2]
    corners = np.array([[x, y, z] for y in ys for z in zs for x in xs])   # 8
    cp = to_camera(corners, cam["pos"], cam["yaw"], cam["pitch"])
    if (cp[:, 2] <= NEAR).any():
        return None                       # partially behind camera: caller draws an edge arrow
    px = to_pixels(cp, W, H, fov)
    if np.abs(px).max() > 20 * max(W, H):
        return None                       # pig is practically on the camera
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    if fill is not None:                  # painter's algorithm, far faces first
        order = sorted(faces, key=lambda f: -cp[list(f), 2].mean())
        shade = [1.0, 1.15, 0.8, 0.9, 0.7, 0.85]
        for k, f in enumerate(order):
            c = tuple(int(min(255, v * shade[k % 6])) for v in fill)
            draw.polygon([tuple(px[i]) for i in f], fill=c, outline=(0, 0, 0))
    else:
        for f in faces:
            pts = [tuple(px[i]) for i in f] + [tuple(px[f[0]])]
            for a_, b_ in zip(pts[:-1], pts[1:]):
                safe_line(draw, a_, b_, W, H, fill=color, width=width)
    return px


def draw_marker(draw, p_world, cam, W, H, fov, color, r=5, filled=False, label=None):
    cp = to_camera([p_world], cam["pos"], cam["yaw"], cam["pitch"])
    if cp[0, 2] <= NEAR:
        return None
    px = to_pixels(cp, W, H, fov)[0]
    box = [px[0] - r, px[1] - r, px[0] + r, px[1] + r]
    draw.ellipse(box, fill=color if filled else None, outline=color, width=2)
    if label:
        draw.text((px[0] + r + 2, px[1] - r), label, fill=color)
    return px


def edge_arrow(draw, p_world, cam, W, H, fov, color, text=""):
    """Off-screen indicator: arrow on the left/right edge toward the target."""
    cp = to_camera([p_world], cam["pos"], cam["yaw"], cam["pitch"])[0]
    side = 1 if cp[0] >= 0 else -1
    x = W - 6 if side > 0 else 6
    y = H // 2
    draw.polygon([(x, y), (x - 9 * side, y - 7), (x - 9 * side, y + 7)], fill=color)
    if text:
        draw.text((x - 34 if side > 0 else 12, y - 5), text, fill=color)


# ── high-level overlay ───────────────────────────────────────────────────────
GOLD, GOLD_DIM, MEMBER = (230, 190, 90), (150, 125, 60), (255, 235, 170)


def overlay(frame, cam, agent_future, pig_future, members_now=None, pig_now=None,
            fov=DEFAULT_FOV, scale=1, text=None):
    """Draw imagined future on `frame` (H,W,3 uint8 or PIL image) → PIL image.

    cam: {"pos": eye xyz, "yaw": deg, "pitch": deg}
    agent_future / pig_future: (K,3) world feet positions for steps t..t+K-1
    members_now: list of (3,) ensemble guesses for the pig at the current step
    pig_now: (3,) imagined pig position at the current step (ghost box)
    """
    img = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))
    img = img.convert("RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    lw = max(1, scale)

    if agent_future is not None and len(agent_future) > 1:
        draw_polyline(d, agent_future, cam, W, H, fov, GOLD_DIM + (200,), width=2 * lw, dash=3)
        for i, p in enumerate(agent_future[1:], start=1):
            draw_marker(d, p, cam, W, H, fov, GOLD_DIM + (230,), r=2 * lw + 1, filled=True)
    if pig_future is not None and len(pig_future) > 1:
        draw_polyline(d, pig_future, cam, W, H, fov, GOLD + (230,), width=2 * lw)
        for i, p in enumerate(pig_future[1:], start=1):
            if i % 2 == 0:
                draw_marker(d, p, cam, W, H, fov, GOLD + (230,), r=2 * lw + 1, label=str(i) if scale > 1 else None)
    for m in (members_now or []):
        draw_marker(d, m, cam, W, H, fov, MEMBER + (170,), r=2 * lw, filled=True)
    if pig_now is not None:
        px = draw_box(d, pig_now, cam, W, H, fov, GOLD + (255,), width=2 * lw)
        if px is None:
            edge_arrow(d, pig_now, cam, W, H, fov, GOLD + (255,), "pig")
    if text:
        d.rectangle([0, 0, W, 12 * lw], fill=(18, 21, 26, 190))
        d.text((3, 1), text, fill=GOLD + (255,))
    return img


def cam_from_obs(obs, cfg):
    spawn = np.asarray(cfg.SPAWN, dtype=np.float64)
    feet = spawn + np.asarray(obs[6:9], dtype=np.float64)
    return {"pos": feet + np.array([0, EYE_HEIGHT, 0]), "yaw": float(obs[1]) * 180.0,
            "pitch": float(obs[2]) * 90.0}


def overlay_from_obs(frame, real_obs, imag_obs_seq, cfg, members_obs=None, fov=DEFAULT_FOV,
                     scale=1, text=None):
    """LIVE entry point. `frame` = Malmo frame at the current step, `real_obs` =
    the observation at that step, `imag_obs_seq` = imagined obs starting from
    `real_obs` (index 0 = now). `members_obs` = optional (K,T,D) per-member seqs."""
    from evaluation.imagination_vs_reality import decode_positions
    cam = cam_from_obs(real_obs, cfg)
    ag, pg = decode_positions(imag_obs_seq, cfg)
    members_now = None
    if members_obs is not None:
        members_now = [decode_positions(m[:1], cfg)[1][0] for m in members_obs]
    return overlay(frame, cam, ag, pg, members_now=members_now,
                   pig_now=pg[0] if pg is not None else None, fov=fov, scale=scale, text=text)


# ── synthetic stand-in scene (offline testing only) ──────────────────────────
def render_synthetic_frame(cam, pig_feet, arena, W=224, H=168, fov=DEFAULT_FOV):
    """Sky + grass + 1-block grid + arena wall + pig box, from the same camera model.
    Pure stand-in for a Malmo frame — it shares the projection code, so it can only
    show the overlay is self-consistent, not that the camera constants match Minecraft."""
    img = Image.new("RGB", (W, H), (135, 175, 220))
    d = ImageDraw.Draw(img)
    lo, hi = arena
    gy = cam["pos"][1] - EYE_HEIGHT
    # no camera roll → the ground plane is exactly everything below the horizon line
    fpx = (H / 2.0) / math.tan(math.radians(fov) / 2.0)
    horizon = H / 2.0 - fpx * math.tan(math.radians(cam["pitch"]))
    if horizon < H:
        d.rectangle([0, max(0.0, horizon), W, H], fill=(86, 110, 58))
    for v in range(int(lo), int(hi) + 2):
        for a, b in (([v, gy, lo], [v, gy, hi + 1]), ([lo, gy, v], [hi + 1, gy, v])):
            seg = _clip_segment(*to_camera([a, b], cam["pos"], cam["yaw"], cam["pitch"]))
            if seg:
                p = to_pixels(np.stack(seg), W, H, fov)
                safe_line(d, p[0], p[1], W, H, fill=(70, 92, 48), width=1)
    wall = [[lo, gy, lo], [hi + 1, gy, lo], [hi + 1, gy, hi + 1], [lo, gy, hi + 1], [lo, gy, lo]]
    for i in range(4):
        seg = _clip_segment(*to_camera([wall[i], wall[i + 1]], cam["pos"], cam["yaw"], cam["pitch"]))
        if seg:
            p = to_pixels(np.stack(seg), W, H, fov)
            safe_line(d, p[0], p[1], W, H, fill=(42, 45, 50), width=3)
    draw_box(d, pig_feet, cam, W, H, fov, None, size=(0.9, 0.9, 0.9), fill=(229, 154, 168))
    return img
