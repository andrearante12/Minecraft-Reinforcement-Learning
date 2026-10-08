"""
visualization/wm_dashboard.py
-----------------------------
Build a single self-contained HTML dashboard that CONFIRMS the world-model
infrastructure by actually running it:

  1. readiness checks   — runs validate.py + the offline smoke tests, probes for
                          the experiment harnesses and for a live Malmo client
  2. real Dreamer run   — for each TARGET_MODE, trains the real `Dreamer` agent
                          (ensemble world model + imagination actor-critic) on
                          data from the OFFLINE stand-in env (utils/sim_hunting.py)
  3. imagined vs real   — freezes the model, imagines a fixed action sequence,
                          replays it in the stand-in env, and scores divergence
                          with evaluation/imagination_vs_reality.py (Experiment 2)

!! The environment here is a kinematic simulator, NOT Minecraft. Everything
!! measured on it validates the plumbing; it says nothing about model quality
!! on real Malmo. The page says so, loudly.

Run (train_env, from malmo/rl):
    python visualization/wm_dashboard.py --out ../../site/wm-dashboard.html
"""

import os
import sys
import json
import time
import shutil
import socket
import argparse
import subprocess
import tempfile
import io
import base64
import numpy as np

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_ROOT)

import torch

from training.configs.hunting_cfg import HuntingCFG
from models.actor_critic import ActorCritic
from algos.dreamer import Dreamer
from utils.sim_hunting import SimHunting, scripted_hunter
from evaluation.imagination_vs_reality import (
    imagine_open_loop, divergence_curve, divergence_horizon, decode_positions)
from visualization.imagination_viz import imagine, imagine_ensemble
from visualization.frame_overlay import (
    cam_from_obs, overlay, render_synthetic_frame, EYE_HEIGHT)

MODES = ["penned", "wandering", "fleeing"]
THRESH = 1.0       # blocks; "prediction still close to reality"


# ── 1. readiness checks ──────────────────────────────────────────────────────
def _run(cmd, timeout=600):
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/miniconda3/bin") + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=RL_ROOT, env=env, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout + p.stderr).strip().splitlines()
        return p.returncode == 0, (out[-1] if out else ""), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return False, repr(e), time.time() - t0


def _port_open(port):
    s = socket.socket()
    s.settimeout(0.3)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def static_checks(skip_slow):
    checks = []

    def add(group, name, status, detail=""):
        checks.append({"group": group, "name": name, "status": status, "detail": detail})

    py = sys.executable
    if skip_slow:
        add("Offline tests", "validate.py / smoke tests", "unverified", "skipped (--skip-checks)")
    else:
        ok, last, dt = _run([py, "utils/validate.py", "--tier", "all", "--env", "hunting", "--quiet"])
        add("Offline tests", "validate.py (static + smoke, env=hunting)",
            "pass" if ok else "fail", "{:.0f}s".format(dt) if ok else last)
        for name, label in [
                ("smoke_video_wm", "Video world model: shapes, loss, KL floor, buffer"),
                ("smoke_dreamer_video", "DreamerVideo: collect→update loop, ckpt round-trip"),
                ("smoke_wire", "Frame wire format + Py3.6 env-server compile gate"),
                ("test_frame_overlay", "First-person overlay: camera projection conventions"),
                ("test_ingame_overlay", "In-game overlay: /particle commands, budget, throttle, dry-run driver")]:
            ok, last, dt = _run([py, "tests/{}.py".format(name)])
            add("Offline tests", label, "pass" if ok else "fail",
                "{:.0f}s".format(dt) if ok else last)

    def exists(rel):
        return os.path.exists(os.path.join(RL_ROOT, rel))

    comp = [
        ("Vector world model (ensemble, probabilistic head)", "models/world_model.py"),
        ("Dreamer agent (imagination actor-critic)", "algos/dreamer.py"),
        ("Hunting env + mission XML", "envs/hunting_env.py"),
        ("Hunting config with uncertainty knobs", "training/configs/hunting_cfg.py"),
        ("Imagination visualiser (fan / uncertainty / overlay)", "visualization/imagination_viz.py"),
        ("Scripted demo generator", "utils/generate_hunting_demos.py"),
    ]
    for name, rel in comp:
        add("Components", name, "pass" if exists(rel) else "missing", rel)

    add("Experiment 2", "Imagined-vs-real divergence harness",
        "pass" if exists("evaluation/imagination_vs_reality.py") else "missing",
        "evaluation/imagination_vs_reality.py (exercised on stand-in env below)")
    add("Experiment 1", "Dreamer vs PPO env-steps-to-threshold comparison script",
        "pass" if exists("evaluation/compare_sample_efficiency.py") else "missing",
        "both agents train via train.py, but nothing yet aggregates the logs")
    add("Experiment 1", "PPO baseline runnable on hunting",
        "pass" if "hunting" in open(os.path.join(RL_ROOT, "training/train.py")).read() else "missing",
        "registered in train.py ENV_REGISTRY")

    # Live
    add("Live Malmo", "Minecraft client reachable on :10000",
        "pass" if _port_open(10000) else "unverified",
        "port open" if _port_open(10000) else "no client running")
    add("Live Malmo", "Hunting demos recorded", "pass" if any(
        os.path.exists(os.path.join(RL_ROOT, "..", "demos", f)) for f in ("hunting.json",)) else "unverified",
        "demos/hunting.json (needs live Malmo)")
    add("Live Malmo", "In-game particle overlay visible (live_imagination.py --probe)", "unverified",
        "needs client + env_server --viz; particle names are 1.11.2 guesses")
    add("Live Malmo", "Chat/slash commands accepted in Survival (op / cheats)", "unverified",
        "if not, the server kicks for spam or ignores /particle")
    add("Live Malmo", "Aiming fix / auto-fire produces kills", "unverified", "first live run")
    add("Live Malmo", "Ensemble Dreamer trained end-to-end on real env", "unverified", "never run")
    return checks


# ── 2. data + training on the stand-in env ───────────────────────────────────
def make_cfg(mode):
    class Cfg(HuntingCFG):
        TARGET_MODE = mode
    return Cfg


def collect(env, cfg, agent, n_steps, rng, eps):
    n = 0
    obs = env.reset()
    while n < n_steps:
        a = scripted_hunter(obs, cfg, rng, eps)
        nxt, r, d, _ = env.step(a)
        agent.buffer.add(obs, a, r, nxt, d)
        obs = env.reset() if d else nxt
        n += 1


def record_episode(env, cfg, rng, eps, max_len=60):
    obs = env.reset()
    obs_seq, acts, rews = [obs], [], []
    for _ in range(max_len):
        a = scripted_hunter(obs, cfg, rng, eps)
        obs, r, d, _ = env.step(a)
        obs_seq.append(obs); acts.append(a); rews.append(r)
        if d:
            break
    return np.array(obs_seq), np.array(acts), np.array(rews, dtype=np.float32)


def xz(p):
    return [[round(float(v[0]), 3), round(float(v[2]), 3)] for v in p]


def feats(obs_seq, cfg):
    p = cfg.PROPRIOCEPTION_SIZE
    o = np.asarray(obs_seq)
    return {
        "dx": o[:, p].round(3).tolist(), "dy": o[:, p + 1].round(3).tolist(),
        "dz": o[:, p + 2].round(3).tolist(),
        "dist": (o[:, p + 6] * cfg.DIST_SCALE).round(2).tolist(),
        "herr": o[:, p + 7].round(3).tolist(), "los": o[:, p + 8].round(2).tolist(),
        "in_range": o[:, p + 9].round(2).tolist(), "life": o[:, p + 10].round(2).tolist(),
        "yaw": (o[:, 1] * 180).round(1).tolist(),
    }


def first_person_frames(cfg, real_obs, imag_obs, member_obs, size):
    """One JPEG (base64) per step: synthetic first-person view from the REAL agent's
    camera with the model's imagined future drawn over it. Base scene is a stand-in
    for a Malmo frame; the overlay path is the same one used live."""
    W, H = size
    ia, ig = decode_positions(imag_obs, cfg)
    _, rg = decode_positions(real_obs, cfg)
    mem = [decode_positions(m, cfg)[1] for m in member_obs]
    out = []
    for t in range(len(real_obs)):
        cam = cam_from_obs(real_obs[t], cfg)
        base = render_synthetic_frame(cam, rg[t], (cfg.ARENA_MIN, cfg.ARENA_MAX), W, H)
        img = overlay(base, cam, ia[t:], ig[t:], members_now=[m[t] for m in mem],
                      pig_now=ig[t], text="t={}  imagined pig + path".format(t))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def run_mode(mode, args, seed):
    torch.manual_seed(seed)
    cfg = make_cfg(mode)
    rng = np.random.RandomState(seed)
    env = SimHunting(cfg, mode, seed=seed)
    agent = Dreamer(ActorCritic(cfg), cfg)
    collect(env, cfg, agent, args.transitions, rng, eps=0.35)
    collect(env, cfg, agent, args.transitions // 3, rng, eps=1.0)   # pure random coverage

    curve, t0 = [], time.time()
    for u in range(1, args.updates + 1):
        logs = agent.update()
        if u % max(1, args.updates // 40) == 0 or u == 1:
            curve.append({"update": u, **{k: round(float(v), 5) for k, v in logs.items()
                                           if isinstance(v, (int, float))}})
    train_s = time.time() - t0

    # checkpoint round-trip
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ck.pt")
    agent.save(path)
    agent2 = Dreamer(ActorCritic(cfg), cfg)
    agent2.load(path)
    probe = torch.as_tensor(np.stack([env.reset() for _ in range(8)]), dtype=torch.float32)
    act = torch.zeros(8, dtype=torch.int64)
    with torch.no_grad():
        same_wm = torch.allclose(agent.world_model.predict(probe, act)[0],
                                 agent2.world_model.predict(probe, act)[0], atol=1e-6)
        same_pi = torch.allclose(agent.model(probe)[0], agent2.model(probe)[0], atol=1e-6)
    shutil.rmtree(tmp, ignore_errors=True)
    if args.save_ckpt_dir:
        agent.save(os.path.join(args.save_ckpt_dir, "dashboard_{}.pt".format(mode)))

    # held-out rollouts: imagine vs real under identical actions
    heval = SimHunting(cfg, mode, seed=seed + 1000)
    rrng = np.random.RandomState(seed + 7)
    rollouts, tries = [], 0
    while len(rollouts) < args.rollouts and tries < 200:
        tries += 1
        obs_seq, acts, rews = record_episode(heval, cfg, rrng, eps=0.3)
        H = args.horizon
        if len(acts) < H:
            continue
        s0 = obs_seq[0]
        acts_h = acts[:H]
        real = obs_seq[:H + 1]
        im = imagine_open_loop(agent.world_model, s0, acts_h)
        ens = imagine_ensemble(agent.world_model, agent.model, s0, H, actions=acts_h, sample=True)
        dream = imagine(agent.world_model, agent.model, s0, H)
        dc = divergence_curve(im["obs_seq"], real, cfg)
        # persistence baseline: "nothing changes"
        pc = divergence_curve(np.repeat(s0[None], H + 1, 0), real, cfg)
        ra, rt = decode_positions(real, cfg)
        ia, it = decode_positions(im["obs_seq"], cfg)
        da, dt_ = decode_positions(dream["obs_seq"], cfg)
        mem = [xz(decode_positions(m, cfg)[1]) for m in ens["per_member"]]
        frames = first_person_frames(cfg, real, im["obs_seq"], ens["per_member"], args.frame_size)
        rollouts.append({
            "id": len(rollouts), "H": H, "frames": frames,
            "actions": [int(a) for a in acts_h],
            "real": {"agent": xz(ra), "pig": xz(rt), "reward": rews[:H].round(3).tolist(),
                     "f": feats(real, cfg)},
            "imag": {"agent": xz(ia), "pig": xz(it), "reward": im["rewards"].round(3).tolist(),
                     "cont": im["conts"].round(3).tolist(), "f": feats(im["obs_seq"], cfg)},
            "members": mem,
            "dream": {"agent": xz(da), "pig": xz(dt_), "actions": [int(a) for a in dream["actions"]],
                      "reward": dream["rewards"].round(3).tolist(),
                      "value": dream["values"].round(2).tolist()},
            "err": {"agent": dc["agent_err"].round(3).tolist(), "pig": dc["target_err"].round(3).tolist(),
                    "max": dc["err"].round(3).tolist(), "persist": pc["err"].round(3).tolist()},
            "ale": ens["aleatoric"].round(5).tolist(), "epi": ens["epistemic"].round(5).tolist(),
            "horizon": divergence_horizon(dc["err"], THRESH),
            "persist_horizon": divergence_horizon(pc["err"], THRESH),
        })

    hz = [r["horizon"] for r in rollouts]
    summary = {
        "n": len(rollouts), "median_horizon": float(np.median(hz)), "mean_horizon": float(np.mean(hz)),
        "median_persist_horizon": float(np.median([r["persist_horizon"] for r in rollouts])),
        "never_diverged": int(sum(h >= args.horizon for h in hz)),
        "mean_err": np.mean([r["err"]["max"] for r in rollouts], 0).round(3).tolist(),
        "mean_persist": np.mean([r["err"]["persist"] for r in rollouts], 0).round(3).tolist(),
        "mean_ale": np.mean([r["ale"] for r in rollouts], 0).round(5).tolist(),
        "mean_epi": np.mean([r["epi"] for r in rollouts], 0).round(5).tolist(),
        "train_seconds": round(train_s, 1), "updates": args.updates,
        "transitions": len(agent.buffer), "ckpt_wm": bool(same_wm), "ckpt_policy": bool(same_pi),
        "n_params_wm": int(sum(p.numel() for p in agent.world_model.parameters())),
    }
    print("[{}] median divergence horizon {} (persistence {}) | train {:.0f}s".format(
        mode, summary["median_horizon"], summary["median_persist_horizon"], train_s))
    return {"curve": curve, "rollouts": rollouts, "summary": summary}


def derived_checks(modes):
    out = []

    def add(name, ok, detail):
        out.append({"group": "Exercised on stand-in env", "name": name,
                    "status": "pass" if ok else "fail", "detail": detail})

    allm = modes.values()
    add("Real Dreamer.update() trains WM + imagination actor-critic (3 modes)",
        all(np.isfinite([c.get("actor_loss", np.nan) for c in m["curve"]]).all() for m in allm),
        "{} updates/mode, losses finite".format(next(iter(allm))["summary"]["updates"]))
    add("World-model checkpoint round-trips",
        all(m["summary"]["ckpt_wm"] and m["summary"]["ckpt_policy"] for m in allm),
        "identical predictions + policy logits after save→load")
    def tail(k, key):
        return float(np.mean(modes[k]["summary"][key][-5:]))
    add("Imagination beats 'nothing changes' baseline (mean error, last 5 steps)",
        all(tail(k, "mean_err") < tail(k, "mean_persist") for k in modes),
        ", ".join("{}: {:.2f} vs {:.2f} blocks".format(k, tail(k, "mean_err"), tail(k, "mean_persist"))
                  for k in modes))
    share = [float(np.mean([c["wm_reward_loss"] / max(c["wm_loss"], 1e-8) for c in m["curve"][-5:]]))
             for m in modes.values()]
    out.append({"group": "Exercised on stand-in env", "name": "World-model loss is balanced across heads",
                "status": "pass" if max(share) < 0.9 else "warn",
                "detail": "reward head = {} of total loss (REWARD_KILL=+100 swamps dynamics heads; "
                          "consider symlog/clipped reward targets)".format(
                              ", ".join("{:.0%}".format(x) for x in share))})
    ale = [float(np.mean(modes[k]["summary"]["mean_ale"])) for k in MODES if k in modes]
    add("Aleatoric uncertainty rises penned → wandering → fleeing",
        all(a < b for a, b in zip(ale, ale[1:])),
        " → ".join("{:.4f}".format(a) for a in ale))
    gate = [float(np.mean([c.get("eff_horizon", 0) for c in modes[k]["curve"][-5:]])) for k in MODES]
    add("Horizon gating shortens effective horizon (eff_horizon < H)",
        all(g < 15 for g in gate), "eff_horizon ≈ " + ", ".join("{:.1f}".format(g) for g in gate))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RL_ROOT, "..", "..", "site", "wm-dashboard.html"))
    ap.add_argument("--transitions", type=int, default=4000)
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--save-ckpt-dir", default=None)
    ap.add_argument("--frame-size", type=int, nargs=2, default=(256, 192))
    ap.add_argument("--refresh-checks", action="store_true", help="with --from-json: re-run readiness checks")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--from-json", default=None, help="skip compute; render HTML from a saved .json")
    args = ap.parse_args()

    if args.from_json:
        data = json.load(open(args.from_json))
        if args.refresh_checks:
            keep = [c for c in data["checks"] if c["group"].startswith("Exercised")]
            data["checks"] = static_checks(False) + keep
        return write(data, args)
    checks = static_checks(args.skip_checks)
    modes = {m: run_mode(m, args, seed=i) for i, m in enumerate(MODES)}
    checks += derived_checks(modes)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RL_ROOT,
                                         text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    data = {"meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "commit": commit,
                     "horizon": args.horizon, "threshold": THRESH, "synthetic": True,
                     "actions": [a[0] for a in HuntingCFG.ACTIONS], "arena": [HuntingCFG.ARENA_MIN,
                                                                               HuntingCFG.ARENA_MAX]},
            "checks": checks, "modes": modes}

    write(data, args)


def write(data, args):
    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm_dashboard_template.html")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.json_only:
        json.dump(data, open(out + ".json", "w"))
    else:
        html = open(tpl).read().replace("/*__DATA__*/null", json.dumps(data))
        open(out, "w").write(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
