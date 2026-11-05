"""
evaluation/latent_probe.py -- the offline gate for the vision world model.

Can the cow be read out of the RSSM latent, and does the reward head pay for
actions regardless of the cow? Runs on CPU or GPU, needs no Malmo: it uses the
demo set (cached DINOv2 tokens + privileged labels) only.

Two modes:
  --checkpoint <pt>   probe an existing checkpoint
  --pretrain          build a fresh agent from the current config, fit the world
                      model on the demos (--wm-steps) and BC the actor
                      (--bc-steps), then probe. This is the gate to pass before
                      any live run: iterate on the config here first.

Held-out demo episodes (--holdout, taken from the end of the index) are never
trained on. Gate criteria (from docs/world_model/results/m2_demo_warmstart):
  bearing R2 >= 0.5, in_range R2 >= 0.3, attack reward out of range <= 25% of
  in range, BC greedy teacher agreement >= 50%.

    python evaluation/latent_probe.py --pretrain --demo-path ../demos/hunting_vision_L0_m3 [--fair] [--cfg K=V]
    python evaluation/latent_probe.py --checkpoint checkpoints/<ckpt>.pt --demo-path ... [--cfg VWM_REWARD_ON_NEXT=False]
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
from training.configs.hunting_vision_cfg import HuntingVisionCFG
from models.video_world_model import privileged_targets
from models.latent_actor_critic import LatentActorCritic
from algos.dreamer_video import DreamerVideo

ACT = ["fwd", "back", "strafeL", "strafeR", "sprint", "turnL", "turnR", "lookUp", "lookDown",
       "attack", "noop", "jumpFwd"]


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]; Xte = np.c_[Xte, np.ones(len(Xte))]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    P = Xte @ W
    ss = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return 1 - ((Yte - P) ** 2).sum(0) / np.maximum(ss, 1e-8)


def attach_priv_labels(agent, demo_path):
    """The demo buffer only carries `priv` when the aux head is on; the probe
    needs the labels either way, so re-read them from the episode files."""
    from utils.demo_io import load_index
    only = getattr(agent.cfg, "DEMO_ONLY_SUCCESS", False)
    entries = [e for e in load_index(demo_path)["episodes"] if not only or e.get("outcome") == "killed"]
    eps = list(agent.demo_buffer.episodes)
    assert len(entries) == len(eps), (len(entries), len(eps))
    for e, ent in zip(eps, entries):
        with np.load(os.path.join(demo_path, ent["file"])) as z:
            e["priv"] = list(z["privileged"].astype(np.float32))


def split_holdout(agent, k):
    """Pop the last k demo episodes out of the agent's demo buffer (never trained on)."""
    eps = agent.demo_buffer.episodes
    test = [eps.pop() for _ in range(min(k, len(eps) - 1))][::-1]
    agent.demo_buffer._total_steps = sum(len(e["actions"]) for e in eps)
    return list(eps), test


def episode_tensors(agent, e):
    dev = agent.device
    f = torch.as_tensor(np.stack(e["feats"]), device=dev).unsqueeze(0)
    v = torch.as_tensor(np.stack(e["vecs"]), dtype=torch.float32, device=dev).unsqueeze(0)
    a = torch.as_tensor(np.asarray(e["actions"]), dtype=torch.int64, device=dev)
    return f, v, a


@torch.no_grad()
def run_episodes(agent, E):
    wm, cfg = agent.world_model, agent.cfg
    out = dict(post=[], prior=[], kl=[], rhat=[], pact=[], priv=[], act=[], rew=[])
    for e in E:
        f, v, a = episode_tensors(agent, e)
        oh = F.one_hot(a, cfg.N_ACTIONS).float().unsqueeze(0)
        embed, _ = wm._embed(f, v)
        po, pr = wm.rssm.observe(embed, wm._shift_actions(oh))
        fp, fr = wm.rssm.get_feat(po)[0], wm.rssm.get_feat(pr)[0]
        out["post"].append(fp.cpu().numpy()); out["prior"].append(fr.cpu().numpy())
        out["kl"].append(wm.rssm.kl_loss(po, pr)[0].cpu().numpy())
        head_in = wm._head_input_seq(wm.rssm.get_feat(po), po, pr, oh)[0]
        r, _ = wm.predict_reward_cont(head_in, oh[0]) if not wm.reward_on_next else wm.predict_reward_cont(head_in)
        out["rhat"].append(r.cpu().numpy())
        st = {k: v_[0] for k, v_ in po.items()}
        allr = []
        for k in range(cfg.N_ACTIONS):
            o = torch.zeros(len(fp), cfg.N_ACTIONS, device=agent.device); o[:, k] = 1
            allr.append(wm.step_reward_cont(st, o)[0].cpu().numpy())
        out["pact"].append(np.stack(allr, 1))
        out["priv"].append(np.stack(e["priv"]) if "priv" in e else np.zeros((len(fp), 12), np.float32))
        out["act"].append(np.asarray(e["actions"])); out["rew"].append(np.asarray(e["rewards"], np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def probe(agent, train_eps, test_eps, label="", live_data=False):
    cfg = agent.cfg
    Tr, Te = run_episodes(agent, train_eps), run_episodes(agent, test_eps)
    Ytr, Mtr = privileged_targets(torch.as_tensor(Tr["priv"])); Yte, Mte = privileged_targets(torch.as_tensor(Te["priv"]))
    Ytr, Mtr, Yte, Mte = Ytr.numpy(), Mtr.numpy() > 0.5, Yte.numpy(), Mte.numpy() > 0.5
    print("\n=== probe %s  (train %d eps / held-out %d eps, %d sensed test steps)" % (
        label, len(train_eps), len(test_eps), int(Mte.sum())))
    r2p = ridge_r2(Tr["post"][Mtr], Ytr[Mtr], Te["post"][Mte], Yte[Mte], alpha=10.0)
    r2r = ridge_r2(Tr["prior"][Mtr], Ytr[Mtr], Te["prior"][Mte], Yte[Mte], alpha=10.0)
    print("posterior feat  R2 sin/cos/dist/in_range = %s" % np.round(r2p, 2))
    print("1-step prior    R2 sin/cos/dist/in_range = %s" % np.round(r2r, 2))
    print("raw KL(post||prior)/step: mean %.2f  p90 %.2f  (free nats %.1f)" % (
        Te["kl"].mean(), np.percentile(Te["kl"], 90), cfg.WM_FREE_NATS))
    c = np.corrcoef(Te["rhat"], Te["rew"])[0, 1]
    print("reward head on held-out demos: pred mean %+.3f vs actual %+.3f, corr %.2f" % (
        Te["rhat"].mean(), Te["rew"].mean(), c))
    inr = Yte[:, 3] > 0.5
    pa = Te["pact"]
    att_in, att_out = pa[inr, 9].mean(), pa[~inr, 9].mean()
    print("predicted reward after each action | cow IN range (n=%d) / OUT (n=%d):" % (inr.sum(), (~inr).sum()))
    print("   " + "  ".join("%s %+.2f/%+.2f" % (ACT[k], pa[inr, k].mean(), pa[~inr, k].mean()) for k in range(cfg.N_ACTIONS)))
    with torch.no_grad():
        logits, val = agent.model(torch.as_tensor(Te["post"], device=agent.device))
        p = torch.softmax(logits, -1).cpu()
        ent = -(p * p.clamp_min(1e-8).log()).sum(-1).mean().item()
        agree = (p.argmax(-1).numpy() == Te["act"]).mean()
    print("actor on held-out demo states: greedy teacher agreement %.0f%%, entropy %.2f, mean V %.1f" % (
        100 * agree, ent, val.mean().item()))
    # open-loop imagination from held-out posteriors
    with torch.no_grad():
        wm = agent.world_model
        f, v, a = episode_tensors(agent, test_eps[0])
        oh = F.one_hot(a, cfg.N_ACTIONS).float().unsqueeze(0)
        embed, _ = wm._embed(f, v)
        po, _ = wm.rssm.observe(embed, wm._shift_actions(oh))
        for mode in ("actor", "random"):
            st = {k: v_[0].clone() for k, v_ in po.items()}; rs = []
            for t in range(15):
                feat = wm.rssm.get_feat(st)
                if mode == "actor":
                    act = torch.distributions.Categorical(logits=agent.model(feat)[0]).sample()
                else:
                    act = torch.randint(0, cfg.N_ACTIONS, (len(feat),), device=agent.device)
                o = F.one_hot(act, cfg.N_ACTIONS).float()
                r, _, st = wm.step_reward_cont(st, o, pessimism=float(getattr(cfg, "IMAG_REWARD_PESSIMISM", 0.0)))
                rs.append(r.mean().item())
            print("imagined reward/step from held-out demo starts, %-6s policy: mean %+.2f (t0 %+.2f, t14 %+.2f)" % (
                mode, np.mean(rs), rs[0], rs[14]))
    ratio = att_out / max(att_in, 1e-6)
    gate = dict(bearing_r2=float(min(r2p[0], r2p[1])), in_range_r2=float(r2p[3]),
                attack_out_over_in=float(ratio), teacher_agreement=float(agree))
    # The attack in/out criterion needs real (agent) data: the teacher never
    # attacks out of range, so a demo-only model has no counterexamples.
    judge_attack = live_data
    ok = (gate["bearing_r2"] >= 0.5 and gate["in_range_r2"] >= 0.3 and agree >= 0.5
          and (ratio <= 0.25 or not judge_attack))
    print("GATE %s  bearing_r2=%.2f (>=0.5)  in_range_r2=%.2f (>=0.3)  attack out/in=%.2f (%s)  agreement=%.2f (>=0.5)" % (
        "PASS" if ok else "FAIL", gate["bearing_r2"], gate["in_range_r2"], ratio,
        "<=0.25" if judge_attack else "reported only: demo-only model, no real negatives", agree))
    return gate, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-path", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--wm-steps", type=int, default=2000)
    ap.add_argument("--bc-steps", type=int, default=2000)
    ap.add_argument("--holdout", type=int, default=15)
    ap.add_argument("--fair", action="store_true")
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--save", default=None, help="save the pretrained agent as a checkpoint")
    a = ap.parse_args()
    from training.train import apply_cfg_overrides, apply_fair_settings
    cfg = HuntingVisionCFG
    cfg.DEMO_PATH = a.demo_path
    if a.fair:
        apply_fair_settings(cfg, "dreamer_video", keep_demos=True)
    apply_cfg_overrides(cfg, a.cfg)
    torch.manual_seed(0); np.random.seed(0)
    agent = DreamerVideo(LatentActorCritic(cfg), cfg)
    attach_priv_labels(agent, a.demo_path)
    train_eps, test_eps = split_holdout(agent, a.holdout)
    if a.checkpoint:
        agent.load(a.checkpoint)
        label = os.path.basename(a.checkpoint)
    elif a.pretrain:
        agent.pretrain_world_model(a.wm_steps)
        agent.pretrain_actor_bc(a.bc_steps)
        label = "pretrained wm %d / bc %d" % (a.wm_steps, a.bc_steps)
        if a.save:
            agent.save(a.save)
    else:
        raise SystemExit("pass --checkpoint or --pretrain")
    agent.world_model.eval(); agent.model.eval()
    probe(agent, train_eps, test_eps, label, live_data=bool(a.checkpoint))


if __name__ == "__main__":
    main()
