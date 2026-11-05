"""visualization/imagined_vs_real.py -- imagined vs real, open-loop, on a recorded episode.

At each step t: RSSM posterior over real steps 0..t, then imagine k steps ahead with
the actions that were actually taken, decode the thumbnail, compare with real t+k.
Writes an mp4: row 1 real (now, +1, +3, +5, +8), row 2 model (recon, imagined ...),
with predicted vs real reward under each future.
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
import subprocess
from PIL import Image, ImageDraw, ImageFont
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
from visualization.dream_strip_viz import load_models
from training.configs.hunting_vision_cfg import HuntingVisionCFG

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--demo", required=True, help="ep_XXXX path prefix (without .npz)")
ap.add_argument("--out", required=True)
ap.add_argument("--cfg", action="append", default=[])
a = ap.parse_args()

cfg = HuntingVisionCFG
from training.train import apply_cfg_overrides, apply_fair_settings
apply_fair_settings(cfg, "dreamer_video")
apply_cfg_overrides(cfg, a.cfg)
wm, model, dev = load_models(a.checkpoint, cfg)

d = np.load(a.demo + ".npz")
f = np.load(a.demo + ".dinov2_vits14_g4_t128.npz")
feats, thumbs = f["feats"], f["thumbs"]
vecs, acts, rews = d["vecs"].astype(np.float32), d["actions"].astype(np.int64), d["rewards"]
L, NA = len(acts), wm.n_actions
KS = [1, 3, 5, 8]
names = [x[0] for x in cfg.ACTIONS]

with torch.no_grad():
    ft = torch.as_tensor(feats, device=dev).unsqueeze(0)
    vt = torch.as_tensor(vecs, device=dev).unsqueeze(0)
    oh = F.one_hot(torch.as_tensor(acts, device=dev).unsqueeze(0), NA).float()
    embed, _ = wm._embed(ft, vt)
    posts, _ = wm.rssm.observe(embed, wm._shift_actions(oh))
    recon, _ = wm.decode(wm.rssm.get_feat(posts))
    recon = (recon.squeeze(0) + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

def img(x):  # float [0,1] or uint8 -> uint8 RGB 256px
    x = (x * 255).astype(np.uint8) if x.dtype != np.uint8 else x
    return np.asarray(Image.fromarray(x).resize((256, 256), Image.NEAREST)).copy()

def label(tile, txt, color=(255, 255, 255), y=22):
    im = Image.fromarray(tile); d = ImageDraw.Draw(im)
    d.text((6, y - 16), txt, font=FONT, fill=color, stroke_width=3, stroke_fill=(0, 0, 0))
    tile[:] = np.asarray(im)
    return tile

blank = np.zeros((256, 256, 3), np.uint8)
W, H = 256 * 5, 256 * 2 + 40
vw = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", "4", "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", a.out], stdin=subprocess.PIPE)
errs = {k: [] for k in KS}
for t in range(L):
    state = {k: v[:, t] for k, v in posts.items()}
    fut, rpred = {}, {}
    with torch.no_grad():
        for k in range(1, max(KS) + 1):
            if t + k > L:
                break
            a1 = F.one_hot(torch.tensor([acts[t + k - 1]], device=dev), NA).float()
            r, _, state = wm.step_reward_cont(state, a1)
            if k in KS:
                im, _ = wm.decode(wm.rssm.get_feat(state))
                fut[k] = (im.squeeze(0) + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                rpred[k] = float(r.item())
    top = [label(img(thumbs[t]), "REAL now  t=%d" % t)]
    bot = [label(img(recon[t]), "model's view")]
    for k in KS:
        if k in fut:
            real_r = float(rews[t + k - 1])
            tk = min(t + k, L - 1); e = float(np.abs(fut[k] - thumbs[tk] / 255.0).mean())
            errs[k].append(e)
            top.append(label(img(thumbs[tk]), "REAL +%d   r=%+.1f" % (k, real_r)))
            col = (120, 255, 120) if abs(rpred[k] - real_r) < 2 else (255, 140, 80)
            bot.append(label(label(img(fut[k]), "IMAGINED +%d" % k, (120, 200, 255)),
                             "pred r=%+.1f" % rpred[k], col, y=244))
        else:
            top.append(blank.copy()); bot.append(blank.copy())
    bar = np.full((40, W, 3), 30, np.uint8)
    kill = "   KILL" if rews[t] >= 50 else ""
    label(bar, "action taken: %s   reward now: %+.1f%s   (imagined frames use the real future actions)"
          % (names[acts[t]], rews[t], kill), y=26)
    vw.stdin.write(np.ascontiguousarray(np.vstack([bar, np.hstack(top), np.hstack(bot)])).tobytes())
vw.stdin.close(); vw.wait()
print("wrote", a.out, "| mean abs pixel error by steps ahead:",
      {k: round(float(np.mean(v)), 3) for k, v in errs.items() if v})
