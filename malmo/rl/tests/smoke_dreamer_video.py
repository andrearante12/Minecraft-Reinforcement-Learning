"""
tests/smoke_dreamer_video.py
------------------------------
Offline (no Malmo) end-to-end smoke test for DreamerVideo, run in train_env:

    conda run -n train_env python malmo/rl/tests/smoke_dreamer_video.py

Uses a FakeVideoEnv (random frames/vecs, mixed episode lengths incl. shorter
than WM_SEQ_LEN) standing in for EnvClient(video=True) + a live Malmo hunting
episode, so the whole collect -> buffer -> update loop can be verified without
Minecraft running.

Checks:
  - collect + update loop runs for enough steps/updates with all logged losses
    finite the whole time.
  - a constant-frame env is learnable: vwm_image trends down (sanity that the
    RSSM + decoder can actually fit something, not just "doesn't crash").
  - per-env RSSM state resets to the initial state on episode done.
  - save() -> fresh agent load() reproduces identical action logits.
  - update() wall-clock, printed for reference (no hard pass/fail threshold —
    hardware-dependent).
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # malmo/rl
sys.path.insert(0, ROOT)

import numpy as np
import torch

from training.configs.hunting_video_cfg import HuntingVideoCFG
from models.latent_actor_critic import LatentActorCritic
from algos.dreamer_video import DreamerVideo


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print("[{0}] {1}".format(status, name))
    if not cond:
        raise SystemExit("Smoke test failed: {0}".format(name))


class FakeVideoEnv:
    """Stands in for EnvClient(video=True): reset()->(vec,frame),
    step(a)->((vec,frame), reward, done, info). Episode lengths vary,
    including some shorter than WM_SEQ_LEN, to exercise buffer padding."""

    def __init__(self, cfg, min_len=4, max_len=25, seed=0, constant_frame=False):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.min_len = min_len
        self.max_len = max_len
        self.constant_frame = constant_frame
        self._fixed_frame = np.full(
            (cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, cfg.VIDEO_CHANNELS), 128, dtype=np.uint8)
        self._ep_len = None
        self._t = 0

    def reset(self):
        self._ep_len = self.rng.randint(self.min_len, self.max_len + 1)
        self._t = 0
        return self._obs()

    def _obs(self):
        vec = self.rng.randn(self.cfg.INPUT_SIZE).astype(np.float32)
        if self.constant_frame:
            frame = self._fixed_frame
        else:
            frame = self.rng.randint(
                0, 256, (self.cfg.VIDEO_HEIGHT, self.cfg.VIDEO_WIDTH, self.cfg.VIDEO_CHANNELS),
                dtype=np.uint8)
        return vec, frame

    def step(self, action):
        self._t += 1
        done = self._t >= self._ep_len
        reward = float(self.rng.randn())
        info = {"outcome": "timeout"}
        return self._obs(), reward, done, info

    def close(self):
        pass


def run_loop(agent, envs, n_steps):
    obs_all = [env.reset() for env in envs]
    for _ in range(n_steps):
        obs_all, rewards, dones, infos = agent.collect_steps(envs, obs_all)
        for i, d in enumerate(dones):
            if d:
                obs_all[i] = envs[i].reset()
    return obs_all


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = HuntingVideoCFG()
    cfg.LEARNING_STARTS = 64
    cfg.VWM_SEQ_BATCH = 8
    cfg.IMAG_BATCH = 32

    n_envs = 2
    envs = [FakeVideoEnv(cfg, seed=i) for i in range(n_envs)]
    model = LatentActorCritic(cfg)
    agent = DreamerVideo(model, cfg, n_envs=n_envs)

    print()
    print("=" * 60)
    print("1. Collect to LEARNING_STARTS + run updates, all losses finite")
    print("=" * 60)
    run_loop(agent, envs, n_steps=cfg.LEARNING_STARTS + 10)
    check("buffer_full() after warmup", agent.buffer_full())

    all_finite = True
    image_losses = []
    for i in range(20):
        logs = agent.update()
        for k, v in logs.items():
            if not np.isfinite(v):
                all_finite = False
                print("  non-finite metric at update {0}: {1}={2}".format(i, k, v))
        if "vwm_image" in logs:
            image_losses.append(logs["vwm_image"])
    check("all logged losses finite across 20 updates", all_finite)
    print("  sample update keys:", sorted(logs.keys()))

    print()
    print("=" * 60)
    print("2. Learnability sanity: constant-frame env -> vwm_image trends down")
    print("=" * 60)
    const_cfg = HuntingVideoCFG()
    const_cfg.LEARNING_STARTS = 64
    const_cfg.VWM_SEQ_BATCH = 8
    const_cfg.IMAG_BATCH = 32
    const_envs = [FakeVideoEnv(const_cfg, seed=i, constant_frame=True) for i in range(2)]
    const_model = LatentActorCritic(const_cfg)
    const_agent = DreamerVideo(const_model, const_cfg, n_envs=2)
    run_loop(const_agent, const_envs, n_steps=const_cfg.LEARNING_STARTS + 10)

    losses = []
    for i in range(60):
        logs = const_agent.update()
        losses.append(logs["vwm_image"])
    early = np.mean(losses[:10])
    late = np.mean(losses[-10:])
    print("  vwm_image early avg: {0:.5f}  late avg: {1:.5f}".format(early, late))
    check("vwm_image decreases on a learnable (constant-frame) env", late < early)

    print()
    print("=" * 60)
    print("3. Per-env RSSM state resets to initial on episode done")
    print("=" * 60)
    fresh_agent = DreamerVideo(LatentActorCritic(cfg), cfg, n_envs=1)
    fresh_env = FakeVideoEnv(cfg, min_len=3, max_len=3, seed=42)
    obs = fresh_env.reset()
    done = False
    while not done:
        obs, reward, done, info = fresh_agent.collect_step(fresh_env, obs, env_idx=0)
    init_state = fresh_agent._initial_state()
    state_after_done = fresh_agent._env_state[0]
    check("deter resets to zeros after done",
          torch.allclose(state_after_done["deter"], init_state["deter"]))
    check("stoch resets to zeros after done",
          torch.allclose(state_after_done["stoch"], init_state["stoch"]))
    check("prev_action resets to zero one-hot after done",
          torch.allclose(fresh_agent._env_prev_act[0], fresh_agent._zero_action()))

    print()
    print("=" * 60)
    print("4. save() -> fresh load() reproduces identical action logits")
    print("=" * 60)
    ckpt_path = "/tmp/smoke_dreamer_video_ckpt.pt"
    agent.save(ckpt_path)

    probe_obs = envs[0].reset()
    vec_t = torch.as_tensor(probe_obs[0], dtype=torch.float32, device=agent.device).unsqueeze(0)
    frame_t = torch.as_tensor(probe_obs[1], device=agent.device).unsqueeze(0)
    state0 = agent.world_model.rssm.initial(1, agent.device)
    prev_act0 = agent._zero_action()
    # RSSM posterior sampling is stochastic (torch.randn_like in _sample); seed
    # identically before each pass so this checks weight equality, not luck.
    torch.manual_seed(123)
    with torch.no_grad():
        post = agent.world_model.encode_step(state0, prev_act0, frame_t, vec_t)
        feat = agent.world_model.rssm.get_feat(post)
        logits_before, _ = agent.model(feat)

    fresh_model = LatentActorCritic(cfg)
    fresh_agent = DreamerVideo(fresh_model, cfg, n_envs=n_envs)
    fresh_agent.load(ckpt_path)
    torch.manual_seed(123)
    with torch.no_grad():
        post2 = fresh_agent.world_model.encode_step(state0, prev_act0, frame_t, vec_t)
        feat2 = fresh_agent.world_model.rssm.get_feat(post2)
        logits_after, _ = fresh_agent.model(feat2)
    check("logits identical after save/load round-trip", torch.allclose(logits_before, logits_after))
    os.remove(ckpt_path)

    print()
    print("=" * 60)
    print("5. update() wall-clock (reference only, hardware-dependent)")
    print("=" * 60)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(10):
        agent.update()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.time() - t0) / 10
    print("  avg update() time: {0:.3f}s  (device={1})".format(elapsed, agent.device))

    print()
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
