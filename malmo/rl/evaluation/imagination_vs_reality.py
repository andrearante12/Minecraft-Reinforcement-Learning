"""
evaluation/imagination_vs_reality.py
------------------------------------
Experiment 2 harness: freeze a trained world model, imagine a fixed action
sequence from a real start state, execute that exact sequence in the real env,
and measure how many steps the imagined trajectory stays close to reality.

Env-agnostic: needs only the config (obs layout) and anything with the
`reset() -> obs` / `step(a) -> (obs, r, done, info)` interface (EnvClient,
HuntingEnv, or the offline simulator in utils/sim_hunting.py).

Positions are decoded from the observation itself, so no env access is needed
for scoring:
    agent = SPAWN + obs[6:9]                (pos-rel-spawn, proprio idx 6..8)
    target = agent + obs[P:P+3]             (target dx,dy,dz, hunting only)
"""

import numpy as np
import torch


def real_open_loop(env, actions, start_obs=None):
    """Execute `actions` in the real env. Returns dict(obs_seq, rewards, dones).

    obs_seq has len(actions)+1 entries (start included) unless the episode
    terminates early, in which case it is truncated at the terminal step.
    """
    obs = env.reset() if start_obs is None else start_obs
    obs_seq, rewards, dones = [np.asarray(obs, dtype=np.float32)], [], []
    for a in actions:
        obs, r, done, _ = env.step(int(a))
        obs_seq.append(np.asarray(obs, dtype=np.float32))
        rewards.append(float(r))
        dones.append(bool(done))
        if done:
            break
    return {"obs_seq": np.array(obs_seq), "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones)}


def imagine_open_loop(world_model, start_obs, actions, device=None):
    """Roll the (frozen) world model under `actions`. Ensemble → mean prediction.

    Returns dict(obs_seq (H+1,D), rewards (H,), conts (H,)).
    """
    device = device or next(world_model.parameters()).device
    s = torch.as_tensor(np.asarray(start_obs, dtype=np.float32), device=device).unsqueeze(0)
    seq, rews, conts = [s.squeeze(0).cpu().numpy()], [], []
    with torch.no_grad():
        for a in actions:
            s, r, c = world_model.predict(s, torch.tensor([int(a)], device=device))
            seq.append(s.squeeze(0).cpu().numpy())
            rews.append(float(r.item()))
            conts.append(float(c.item()))
    return {"obs_seq": np.array(seq, dtype=np.float32),
            "rewards": np.array(rews, dtype=np.float32),
            "conts": np.array(conts, dtype=np.float32)}


def decode_positions(obs_seq, cfg):
    """(agent_xyz (T,3), target_xyz (T,3) or None) from an observation sequence."""
    obs_seq = np.asarray(obs_seq)
    agent = np.asarray(cfg.SPAWN, dtype=np.float32) + obs_seq[:, 6:9]
    if getattr(cfg, "TARGET_MOB", None) is None:
        return agent, None
    p = cfg.PROPRIOCEPTION_SIZE
    return agent, agent + obs_seq[:, p:p + 3]


def divergence_curve(imag_obs, real_obs, cfg):
    """Per-step position error in blocks over the common horizon.

    Returns dict(agent_err (T,), target_err (T,) or None, err (T,)) where `err`
    is the max of the available errors (the quantity thresholded below).
    """
    T = min(len(imag_obs), len(real_obs))
    ia, it = decode_positions(imag_obs[:T], cfg)
    ra, rt = decode_positions(real_obs[:T], cfg)
    a_err = np.linalg.norm(ia - ra, axis=1)
    t_err = np.linalg.norm(it - rt, axis=1) if it is not None else None
    err = a_err if t_err is None else np.maximum(a_err, t_err)
    return {"agent_err": a_err, "target_err": t_err, "err": err}


def divergence_horizon(err, threshold=1.0):
    """Number of steps the prediction stays within `threshold` blocks.

    = index of the first step whose error exceeds the threshold (step 0 is the
    shared start state, error 0). If it never diverges, returns len(err) - 1
    (right-censored: the horizon is at least this long).
    """
    over = np.nonzero(np.asarray(err) > threshold)[0]
    return int(over[0]) - 1 if len(over) else len(err) - 1
