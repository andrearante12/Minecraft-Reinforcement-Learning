"""
tests/smoke_dreamer_vision.py
------------------------------
Offline (no Malmo) checks of the trainer side of hunting_vision — frozen
encoder, feature world model, frames demos, DreamerFD-for-video — in train_env:

    conda run -n train_env python malmo/rl/tests/smoke_dreamer_vision.py           # stub encoder
    conda run -n train_env python malmo/rl/tests/smoke_dreamer_vision.py --real    # + real DINOv2

The stub encoder has the exact DINOv2 ViT-S/14 token layout but random fixed
weights, so everything except "the features mean something" is exercised
without a download. --real additionally runs the real backbone (cached by
torch.hub after the first run).

Checks:
  1. encoder: (N,17,384) fp16, frozen (no trainable params, stays in eval)
  2. buffer: fp16 feats / uint8 thumbs keep their dtype; add_episode works
  3. feature world model: finite loss; the thumbnail loss sends NO gradient
     into the RSSM/encoder (stop-grad); loss falls on a fixed batch
  4. demo_io: write -> reopen -> append -> load; JPEG round-trip error small;
     feature cache reused on the second load
  5. DreamerVideo + demos: demos load, pretrain runs, collect+update loop
     finite with a constant update-log key set, demo starts used, BC coef decays
  6. checkpoint round-trip reproduces logits; a different encoder refuses to load
"""

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # malmo/rl
sys.path.insert(0, ROOT)

import numpy as np
import torch

from training.configs.hunting_vision_cfg import HuntingVisionCFG
from models.frozen_encoder import FrozenEncoder, make_thumbs
from models.video_world_model import VideoWorldModel
from models.latent_actor_critic import LatentActorCritic
from algos.sequence_replay_buffer import SequenceReplayBuffer
from algos.dreamer_video import DreamerVideo
from utils.demo_io import DemoWriter, load_index, load_episode, episode_features


def check(name, cond):
    print("[{0}] {1}".format("OK" if cond else "FAIL", name))
    if not cond:
        raise SystemExit("Smoke test failed: " + name)


class TestCFG(HuntingVisionCFG):
    VIS_ENCODER = "stub"
    LEARNING_STARTS = 60
    IMAG_BATCH = 32
    IMAG_HORIZON = 5
    VWM_SEQ_BATCH = 8
    WM_SEQ_LEN = 12
    VWM_MIN_SEQ = 4
    SEQ_BUFFER_CAPACITY = 2000
    DEMO_PATH = None


class FakeVisionEnv:
    """EnvClient(video=True) stand-in: 224x224 frames, 11-dim proprio, 12 actions."""

    def __init__(self, cfg, seed=0, min_len=6, max_len=30):
        self.cfg, self.rng = cfg, np.random.RandomState(seed)
        self.min_len, self.max_len = min_len, max_len

    def _obs(self):
        frame = self.rng.randint(0, 256, (self.cfg.VIDEO_HEIGHT, self.cfg.VIDEO_WIDTH, 3), dtype=np.uint8)
        return self.rng.randn(self.cfg.INPUT_SIZE).astype(np.float32), frame

    def reset(self):
        self.t, self.n = 0, self.rng.randint(self.min_len, self.max_len + 1)
        return self._obs()

    def step(self, action):
        assert 0 <= action < self.cfg.N_ACTIONS
        self.t += 1
        done = self.t >= self.n
        priv = [float(v) for v in self.rng.uniform(-1, 1, 12)]
        priv[11] = 1.0                                  # target sensed -> aux loss active
        return self._obs(), float(self.rng.randn()), done, {
            "outcome": "killed" if done else "alive", "privileged": priv}

    def close(self):
        pass


def write_demos(demo_dir, cfg, n_eps=4, seed=1):
    env, writer = FakeVisionEnv(cfg, seed=seed, min_len=10, max_len=20), DemoWriter(demo_dir, "hunting_vision")
    for _ in range(n_eps):
        vec, frame = env.reset()
        writer.begin_episode()
        done = False
        while not done:
            a = int(env.rng.randint(cfg.N_ACTIONS))
            (nv, nf), r, done, info = env.step(a)
            writer.add(frame, vec, a, r, done, privileged=info["privileged"])
            vec, frame = nv, nf
        writer.end_episode("killed", source="scripted")
    return writer


def main(real):
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = TestCFG
    tmp = tempfile.mkdtemp(prefix="smoke_vision_")
    try:
        print("1. Frozen encoder")
        names = ["stub"] + (["dinov2_vits14"] if real else [])
        frames = np.random.randint(0, 256, (3, 224, 224, 3), dtype=np.uint8)
        for name in names:
            enc = FrozenEncoder(name, pool_grid=4)
            out = enc(frames)
            check("{0}: tokens (3,17,384) fp16".format(name),
                  tuple(out.shape) == (3, 17, 384) and out.dtype == torch.float16)
            check("{0}: no trainable params".format(name),
                  not any(p.requires_grad for p in enc.parameters()))
            enc.train()
            check("{0}: stays in eval mode after .train()".format(name), not enc.backbone.training)
            check("{0}: finite".format(name), torch.isfinite(out.float()).all().item())
        check("thumbnails 64x64 uint8", make_thumbs(frames).shape == (3, 64, 64, 3))

        print("\n2. Sequence buffer with features")
        enc = FrozenEncoder("stub", pool_grid=4)
        buf = SequenceReplayBuffer(500, 12, 4)
        feats = enc(frames).cpu().numpy()
        thumbs = make_thumbs(frames)
        for t in range(20):
            buf.add_obs({"feats": feats[t % 3], "thumbs": thumbs[t % 3]},
                        np.zeros(11, np.float32), t % 12, 0.0, t == 19)
        buf.add_episode({"feats": np.repeat(feats[:1], 8, 0), "thumbs": np.repeat(thumbs[:1], 8, 0)},
                        np.zeros((8, 11)), np.zeros(8, np.int64), np.zeros(8), np.zeros(8))
        b = buf.sample_sequences(4)
        check("feats keep fp16, thumbs keep uint8",
              b["feats"].dtype == np.float16 and b["thumbs"].dtype == np.uint8)
        check("batch shapes", b["feats"].shape == (4, 12, 17, 384) and b["thumbs"].shape == (4, 12, 64, 64, 3))
        check("obs_keys", sorted(buf.obs_keys()) == ["feats", "thumbs"])

        print("\n3. Feature world model")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wm = VideoWorldModel(cfg).to(dev)
        check("feature WM has no pixel decoder / conv encoder",
              not hasattr(wm, "decoder") and wm.input_kind == "features")
        tens = {k: torch.as_tensor(v, device=dev) for k, v in b.items()}
        args = (tens["feats"], tens["vecs"].float(), tens["actions"].long(), tens["rewards"].float(),
                tens["dones"].float(), tens["mask"].float())
        loss, metrics, _ = wm.loss(*args, thumbs=tens["thumbs"])
        check("loss finite", torch.isfinite(loss).item() and all(np.isfinite(list(metrics.values()))))
        # Thumbnail term alone must not reach the latent.
        wm.zero_grad()
        _, _, _ = wm.loss(*args, thumbs=tens["thumbs"])
        wm.zero_grad()
        recon = wm.thumb_decoder(wm.rssm.get_feat(wm.rssm.observe(
            wm._embed(args[0], args[1])[0], wm._shift_actions(
                torch.nn.functional.one_hot(args[2], cfg.N_ACTIONS).float()))[0]).detach())
        recon.pow(2).mean().backward()
        rssm_grad = sum(p.grad.abs().sum().item() for p in wm.rssm.parameters() if p.grad is not None)
        enc_grad = sum(p.grad.abs().sum().item() for p in wm.encoder.parameters() if p.grad is not None)
        thumb_grad = sum(p.grad.abs().sum().item() for p in wm.thumb_decoder.parameters() if p.grad is not None)
        check("stop-grad: thumbnail loss gives RSSM/encoder zero grad, decoder nonzero",
              rssm_grad == 0.0 and enc_grad == 0.0 and thumb_grad > 0.0)
        opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        first = None
        for i in range(60):
            opt.zero_grad()
            loss, metrics, _ = wm.loss(*args, thumbs=tens["thumbs"])
            loss.backward()
            opt.step()
            first = metrics["vwm_feat"] if first is None else first
        check("vwm_feat falls on a fixed batch ({0:.4f} -> {1:.4f})".format(first, metrics["vwm_feat"]),
              metrics["vwm_feat"] < 0.7 * first)

        print("\n4. demo_io")
        demo_dir = os.path.join(tmp, "demos")
        write_demos(demo_dir, cfg, n_eps=3)
        w2 = write_demos(demo_dir, cfg, n_eps=1, seed=2)          # reopen + append
        idx = load_index(demo_dir)
        check("index has 4 episodes after append", len(idx["episodes"]) == 4 == len(w2))
        check("episode files numbered uniquely", len(set(e["file"] for e in idx["episodes"])) == 4)
        ep = load_episode(demo_dir, idx["episodes"][0])
        check("episode arrays aligned", len(ep["frames"]) == len(ep["actions"]) == len(ep["vecs"])
              == len(ep["privileged"]) == idx["episodes"][0]["steps"])
        smooth = np.full((224, 224, 3), 100, np.uint8)
        smooth[:, 112:] = 180
        wj = DemoWriter(os.path.join(tmp, "jpeg"), "x")
        wj.add(smooth, np.zeros(11), 0, 0.0, True)
        wj.end_episode("x")
        back = load_episode(os.path.join(tmp, "jpeg"), load_index(os.path.join(tmp, "jpeg"))["episodes"][0])["frames"][0]
        check("JPEG round-trip mean abs error < 3", np.abs(back.astype(int) - smooth).mean() < 3)
        f1, t1 = episode_features(demo_dir, idx["episodes"][0], enc)
        n_files = len(os.listdir(demo_dir))
        f2, _ = episode_features(demo_dir, idx["episodes"][0], enc)
        check("feature cache written once and reused",
              np.array_equal(f1, f2) and len(os.listdir(demo_dir)) == n_files and f1.dtype == np.float16)

        print("\n5. DreamerVideo (features) + demos")
        cfg.DEMO_PATH = demo_dir
        agent = DreamerVideo(LatentActorCritic(cfg), cfg, n_envs=2)
        check("demo buffer loaded (4 episodes)", agent.demo_buffer is not None and len(agent.demo_buffer.episodes) == 4)
        agent.pretrain_world_model(10, log_every=10)
        envs = [FakeVisionEnv(cfg, seed=10), FakeVisionEnv(cfg, seed=11)]
        obs = [e.reset() for e in envs]
        key_sets, n_updates = set(), 0
        for _ in range(120):
            obs, _, dones, _ = agent.collect_steps(envs, obs)
            for i, d in enumerate(dones):
                if d:
                    obs[i] = envs[i].reset()
            if agent.buffer_full():
                logs = agent.update()
                n_updates += 1
                key_sets.add(tuple(logs))
                check_finite = all(np.isfinite(v) for v in logs.values())
                if not check_finite:
                    check("update logs finite: {0}".format(logs), False)
        check("{0} updates ran, all finite".format(n_updates), n_updates > 10)
        check("update-log key set constant (fixed CSV header)", len(key_sets) == 1)
        check("bc_loss logged and > 0", "bc_loss" in logs and logs["bc_loss"] > 0)
        check("real buffer stores feats+thumbs+priv, not frames",
              sorted(agent.buffer.obs_keys()) == ["feats", "priv", "thumbs"])
        check("privileged aux loss active and finite ({0:.4f})".format(logs["vwm_priv"]),
              logs["vwm_priv"] > 0 and np.isfinite(logs["vwm_priv"]))
        agent.set_progress(0.5)
        check("BC coef decays with progress", abs(agent._demo_bc_coef - 0.5 * cfg.DEMO_BC_COEF) < 1e-9)
        batch, n_demo = agent._sample_batch(cfg.VWM_SEQ_BATCH)
        check("WM batch mixes demo rows (n_demo={0} of {1})".format(n_demo, cfg.VWM_SEQ_BATCH),
              0 < n_demo < cfg.VWM_SEQ_BATCH)

        print("\n6. Checkpoint")
        path = os.path.join(tmp, "ckpt.pt")
        agent.save(path)
        cfg.DEMO_PATH = None
        fresh = DreamerVideo(LatentActorCritic(cfg), cfg)
        fresh.load(path)
        o = envs[0].reset()
        agent.reset_eval_state()
        fresh.reset_eval_state()
        torch.manual_seed(1)
        a1 = agent.select_action(o, greedy=True)
        torch.manual_seed(1)
        a2 = fresh.select_action(o, greedy=True)
        feat1 = agent.world_model.rssm.get_feat(agent._eval_state)
        feat2 = fresh.world_model.rssm.get_feat(fresh._eval_state)
        l1, _ = agent.model(feat1)
        l2, _ = fresh.model(feat2)
        check("round-trip: same greedy action + logits", a1 == a2 and torch.allclose(l1, l2, atol=1e-5))

        class OtherEncCFG(TestCFG):
            VIS_POOL_GRID = 2
        try:
            DreamerVideo(LatentActorCritic(OtherEncCFG), OtherEncCFG).load(path)
            refused = False
        except (ValueError, RuntimeError):
            refused = True
        check("checkpoint from a different encoder config refuses to load", refused)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL DREAMER VISION CHECKS PASSED")


if __name__ == "__main__":
    main(real="--real" in sys.argv)
