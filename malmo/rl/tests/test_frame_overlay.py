"""Offline checks of the camera model in visualization/frame_overlay.py."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.frame_overlay import project, camera_basis, cam_from_obs, overlay_from_obs
from training.configs.hunting_cfg import HuntingCFG as C

W = H = 100
cam = [0.0, 1.62, 0.0]
def px(p, yaw=0, pitch=0): return project([p], cam, yaw, pitch, W, H)[0]

# straight ahead (+Z at yaw 0) → image centre
assert np.allclose(px([0, 1.62, 5]), [50, 50]), px([0, 1.62, 5])
# at yaw 0 facing +Z, +X is to the LEFT, -X to the right (Minecraft)
assert px([2, 1.62, 5])[0] < 50 < px([-2, 1.62, 5])[0]
# higher world y → higher on screen (smaller py)
assert px([0, 3.0, 5])[1] < 50
# ground ahead appears below centre; farther ground is closer to horizon
assert px([0, 0, 3])[1] > px([0, 0, 10])[1] > 50
# behind camera → NaN
assert np.isnan(px([0, 1.62, -3])).all()
# positive pitch looks down: the horizon moves UP the image
assert px([0, 1.62, 5], pitch=30)[1] < 50
# turning (yaw +90 faces -X): a point at -X is dead ahead
assert np.allclose(px([-5, 1.62, 0], yaw=90), [50, 50], atol=1e-6)
# fov: point at half-fov above the axis lands on the top edge
z = 5.0; dy = z * math.tan(math.radians(35))
assert abs(px([0, 1.62 + dy, z])[1] - 0) < 1e-6
# basis is orthonormal & right-handed (up = r × f)
f, r, u = camera_basis(37, 12)
assert abs(np.dot(f, r)) < 1e-9 and abs(np.dot(f, u)) < 1e-9 and abs(np.dot(r, u)) < 1e-9 and u[1] > 0
# cam_from_obs: yaw/pitch normalisation round-trips
o = np.zeros(C.INPUT_SIZE, np.float32); o[1] = 0.5; o[2] = -1 / 3; o[6:9] = [1, 0, 2]
c = cam_from_obs(o, C); assert abs(c["yaw"] - 90) < 1e-4 and abs(c["pitch"] + 30) < 1e-4
assert np.allclose(c["pos"], np.array(C.SPAWN) + [1, 1.62, 2])
# end-to-end live entry point runs on a blank frame
seq = np.stack([o] * 4); 
img = overlay_from_obs(np.zeros((64, 64, 3), np.uint8), o, seq, C)
assert img.size == (64, 64)
print("ALL FRAME-OVERLAY CHECKS PASSED")
