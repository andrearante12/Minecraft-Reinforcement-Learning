# World Models for MalmoRL — Uncertainty-Aware Imagination (Hunting)

*Status: 2026-07-02 · branch `extension/world-model-hunting` · Phase 1 (aiming fix) + Phase 2 (DreamerFD demos) + Phase 3 (hunting_wild) implemented and offline-validated.*

This document explains **what we currently have** and **what we are trying to do**. It is the entry point for the world-model line of work; the task-specific env notes live alongside the parkour/bridging reports.

---

## 1. The one-paragraph thesis

MalmoRL observations are already compact **state vectors** (not pixels), and every Malmo step is a blocking ~0.15 s hold. That makes a **Dreamer-style world model** — a learned dynamics model the agent trains inside via "imagination" — both cheap (a small MLP, no image VAE) and valuable (imagined steps replace expensive real ones). We showcase it on a **hunting** task (locate → aim → approach → attack a pig until it dies) and turn it into a research contribution by attacking one specific weakness of naive imagination: it produces a **single point prediction** of where the pig will go and **trusts it equally at every step**. We instead give the agent an **ensemble of probabilistic world models** so it can *measure* its own uncertainty and *split* it into:

- **aleatoric** — irreducible noise (the pig's un-actioned motion; grows `penned → wandering → fleeing`), and
- **epistemic** — model ignorance (shrinks with data),

then **act on each differently**: shorten imagination where the future is genuinely unpredictable (aleatoric/total → *horizon gating*), and explore toward what is merely unlearned (epistemic → *intrinsic reward*, never aleatoric — the "noisy-TV" guard). The `TARGET_MODE` knob is the controlled variable that makes the split falsifiable, and the same quantity becomes the headline figure (the ensemble's imagined pig-futures **fanning out**).

---

## 2. What we currently have (built + verified offline)

### 2.1 The agent and task (Phases 0–1)
- **`algos/dreamer.py`** — `Dreamer(BaseAgent)`. Collects real transitions → fits the world model → trains the `ActorCritic` almost entirely in imagination (REINFORCE actor + λ-return critic). Policy checkpoints stay byte-compatible with PPO/DQN/BC; the world model persists via `_extra_state()`.
- **`envs/hunting_env.py` + `envs/hunting/missions/hunting.xml`** — walled Survival arena, a per-episode `DrawEntity` pig, 98-dim observation (see §4), 11 discrete actions incl. `attack`, dense reward (target life-drop + approach + aim) with a dominant terminal `REWARD_KILL`. `TARGET_MODE ∈ {penned, wandering, fleeing}` sets target predictability.
- **`models/actor_critic.py`** — unchanged 3-stream policy/value net (proprio / target / voxel).

### 2.2 The uncertainty mechanism (this phase)
- **`models/world_model.py`**
  - `WorldModel` — obs-space dynamics: proprio/goal predicted as **residual deltas**, voxel as occupancy logits, plus reward and continuation heads. Optional **probabilistic goal head** (`WM_PROBABILISTIC`): emits `(mean, log_var)` for the target stream, trained with **Gaussian NLL** (clamped, bias-initialised log-var so an untrained model reports modest — not inflated — aleatoric).
  - `WorldModelEnsemble` — an `nn.ModuleList` of **K** members. `predict` (ensemble mean, drop-in), `predict_all` (per-member), and `predict_with_uncertainty → (next, reward, cont, aleatoric, epistemic)` where `aleatoric = mean_k(var_k)` and `epistemic = var_k(mean_k)` over the pig-position dims. Being one `nn.Module`, it checkpoints and optimises exactly like the single model.
- **`algos/dreamer.py`** consumers:
  - **Bootstrap training** — each member fits its own independently-sampled minibatch (→ real disagreement / nonzero epistemic).
  - **Horizon gating** — trust factor `τ_t = exp(−β · Σ uncertainty)` folded into the `γ·continuation` discount, so diverged far-future dreams stop inflating value targets.
  - **Epistemic-only intrinsic reward** (`INTRINSIC_COEF`, EMA-normalised) — curiosity toward *unlearned* dynamics, never *noisy* ones.
  - New per-update log columns: `aleatoric`, `epistemic`, `eff_horizon`, `intrinsic_r`.

### 2.3 The visual instrument (this phase)
`visualization/imagination_viz.py` (`--mode` CLI):
- **`figure_dream_fan`** — every member's imagined pig path from one start; the fan widens with horizon and with target stochasticity (the "hedge").
- **`figure_uncertainty`** / **`figure_uncertainty_by_mode`** — aleatoric vs epistemic over the horizon, and the cross-mode **knob figure**.
- **`figure_counterfactual`** — one start, several candidate action sequences, branching agent+pig futures.

### 2.4 Verification already done (no Malmo, in `train_env`)
`validate.py` Tier-1 (static) and Tier-2 (smoke) both **PASS**, incl. Dreamer-ensemble instantiation on hunting's 98-dim obs. Offline smoke scripts confirm: ensemble trains with finite NLL; **epistemic shrinks** on repeated fits (reducible) while aleatoric is well-formed; the deterministic path is unchanged and emits stable zero-valued keys; **horizon gating truncates** (`eff_horizon ≈ 3.3 < H=15` at cold start); ensemble checkpoints round-trip; all four figures render and the fan **widens** with horizon.

---

## 3. What we are trying to do next — Phase D (the study)

The deliverable is a controlled study on **live Malmo**, run entirely by editing knobs in `training/configs/hunting_cfg.py` (the single source of truth both processes read) and swapping `--algo`.

| Experiment | What changes | The claim under test |
|---|---|---|
| **Knob study** | `TARGET_MODE` = penned → wandering → fleeing | **aleatoric rises monotonically; epistemic stays ~flat** (it tracks data coverage, not the pig's behaviour) |
| **Gating ablation** | `WM_HORIZON_GATING` on/off under `fleeing` | gating recovers value-estimation quality that fixed-horizon imagination loses when the target is unpredictable |
| **Exploration ablation** | `INTRINSIC_COEF` 0.1 vs 0.0 | epistemic-only intrinsic reward reaches **first kill in fewer real steps** |
| **Headline** | `--algo dreamer` vs `ppo` vs `dqn` | the world-model agent crosses a success threshold in materially fewer real Malmo steps / wall-clock |

**Headline figures** come straight from the instrument: the dream-fan and the cross-mode decomposition.

### 3.1 Phase-D runbook (3 terminals)

```
# T1 — Minecraft client (Malmo)
cd malmo/Minecraft && ./launchClient.sh -port 10000

# T2 — env server (conda: malmo). TARGET_MODE is read HERE (spawn logic runs in-env).
conda run -n malmo python malmo/rl/envs/env_server.py --env hunting --port 9999 --malmo-port 10000

# T3 — training (conda: train_env)
conda run -n train_env python malmo/rl/training/train.py --algo dreamer --env hunting --base-port 9999
#   baseline:  --algo ppo   (repeat for dqn)
```

Notes:
- Learning starts after `LEARNING_STARTS=1000` real transitions (~5 episodes at `MAX_STEPS=200`); before that you'll see collection only.
- Per-run logs are timestamped in `malmo/rl/logs/` (`episodes`, `updates`, `trajectories`). The new uncertainty columns land in `*_updates.csv` automatically.
- To render dreams from a checkpoint:
  `conda run -n train_env python malmo/rl/visualization/imagination_viz.py --env hunting --checkpoint <ckpt.pt> --mode fan` (also `uncertainty`, `counterfactual`).

### 3.2 The one calibration to do first
`WM_TRUST_BETA` (default 0.5) sets how aggressively gating shortens the horizon. Calibrate it once on `penned` so a *trained* model keeps a usefully long `eff_horizon` (watch the `eff_horizon` column). **Keep it an absolute scale — do NOT normalise uncertainty per-run**, or the cross-mode aleatoric signal (the entire finding) would be normalised away.

---

## 4. Reference — observation layout & config knobs

**Observation (98-dim)** — `[proprio(11) | target(12) | voxel(75)]`:
`[0]` onGround `[1]` yaw `[2]` pitch `[3–5]` vel(y,x,z) `[6–8]` pos-rel-spawn `[9]` food `[10]` health · `[11–13]` **target Δ(x,y,z)** ← the pig-position dims uncertainty is measured on · `[14–16]` target vel · `[17]` dist `[18]` heading-err `[19]` los_hit `[20]` in_range `[21]` life `[22]` visible · `[23:98]` 5×3×5 voxel grid.

**Uncertainty knobs** — off by default in `configs/world_model_cfg.py` (protects parkour/bridging + Phase-0), enabled in `configs/hunting_cfg.py`:

| Knob | Hunting value | Meaning |
|---|---|---|
| `ENSEMBLE_SIZE` | 5 | number of world-model members |
| `WM_PROBABILISTIC` | True | Gaussian goal head → aleatoric |
| `WM_UNCERTAINTY_DIMS` | (0,1,2) | target Δx,Δy,Δz within the target stream |
| `WM_HORIZON_GATING` / `WM_TRUST_BETA` | True / 0.5 | trust-factor horizon gating |
| `INTRINSIC_COEF` / `INTRINSIC_KIND` | 0.1 / epistemic | exploration reward (never aleatoric) |
| `WM_MIN/MAX/INIT_LOGVAR` | −8 / 2 / −2 | NLL stability + cold-start calibration |

---

---

## 5. What was added in the sample-efficiency + realism pass (2026-07-02)

### 5.1 Phase 1 — Aiming fix (the hard-exploration bottleneck)
**Problem:** `attack` only deals damage if `los_hit` (ray exactly on the pig), making the aim-AND-attack conjunction unreachable by random exploration.

**Fix in `envs/hunting_env.py`:**
- **Potential-based aim-alignment shaping** (Ng 1999, policy-invariant): `Φ(s) = cos(heading_error·π)`, shaped reward = `AIM_ALIGN_COEF · (γΦ' − Φ)`. Adds zero net discounted reward — no policy bias — but gives a dense gradient toward reducing crosshair-to-pig angle. Heading error is already in obs[18]; `_prev_phi` tracked as instance state.
- **Auto-fire**: when `in_range` and `abs(heading_error) < AUTO_ATTACK_ANGLE (~27°)`, an attack command is automatically fired between steps. Dissolves the conjunction bottleneck.

**New `hunting_cfg.py` knobs:** `AIM_ALIGN_COEF=0.3`, `AUTO_ATTACK=True`, `AUTO_ATTACK_ANGLE=0.15`.

**Verify:** first live run should show `outcome=killed` within tens of episodes. `heading_error` should trend toward 0 as training progresses.

### 5.2 Phase 2 — DreamerFD demo integration
**Goal:** reach first reliable kills in fewer real Malmo steps by leveraging a small expert demonstration set.

**Changes:**
- **`algos/dreamer.py`**: demo buffer loaded from `cfg.DEMO_PATH` on init (`_load_demo_buffer` handles both full-transition and BC-only legacy formats). In `_train_world_model`, `DEMO_WM_FRACTION` (10%) of each WM minibatch comes from the demo buffer — expert transitions teach the world model dynamics it can't see from random exploration. In `_imagine_and_train`, `DEMO_START_FRACTION` (30%) of imagination start states come from the demo buffer — roots imagination in expert territory. `_demo_bc_update()` adds a cross-entropy loss against demo (obs, action) pairs, weighted by `_demo_bc_coef` which decays linearly with training progress so the agent can eventually surpass the demonstrator.
- **`configs/world_model_cfg.py`**: `DEMO_PATH=None`, `DEMO_WM_FRACTION=0.1`, `DEMO_START_FRACTION=0.3`, `DEMO_BC_COEF=0.0` (enable per-run via CLI or override in cfg).
- **`utils/generate_hunting_demos.py`** (new): scripted hunter using obs[17-22] heuristics — turn toward pig, move forward when far, attack when aligned+in_range. Records full `(obs, action, reward, next_obs, done)` transitions. Run 20–50 episodes; successful kills give the clearest signal.
- **`utils/record_demos.py`**: added hunting/hunting_wild to ENV_CONFIGS, `translate_keys_to_action_hunting()`, full transitions now saved in all envs.

**Usage:**
```bash
# Step 1 — generate scripted demos (needs live Malmo + env_server)
conda run -n train_env python malmo/rl/utils/generate_hunting_demos.py --port 9999 --episodes 30

# Step 2 — BC warm-start (optional but cheap)
conda run -n train_env python malmo/rl/training/train.py --algo bc --env hunting --demo-path demos/hunting.json

# Step 3 — Dreamer + demos
conda run -n train_env python malmo/rl/training/train.py --algo dreamer --env hunting --demo-path demos/hunting.json
# (add --checkpoint <bc_ckpt.pt> for BC warm-start)
```

### 5.3 Phase 3 — Generic constrained world (hunting_wild)
**Goal:** show the method scales to realistic terrain, not just a superflat arena.

**Changes:**
- **`envs/hunting_wild/missions/hunting_wild.xml`** (new): `DefaultWorldGenerator seed="4837" forceReset="true"` for reproducible natural plains terrain + a bedrock cage (x=±21, z=±21, y=61–80) bounding a 42×42 play area. No interior clearing — agent navigates real terrain.
- **`configs/hunting_wild_cfg.py`** (new): inherits `HuntingCFG`; overrides `SPAWN=(0.5,65.0,0.5)`, `FALL_Y_THRESHOLD=60.0`, larger `ARENA_MIN/MAX=±18`, expanded `BLOCK_ENCODING` with natural blocks (grass/dirt/stone/gravel/sand/log/leaves/bedrock).
- **Registered in both** `env_server.py` and `train.py` as `"hunting_wild"`.

**Calibration step (one-time):** start env_server on `hunting_wild`, run one episode, check `YPos` in logs. If agent spawns in air (falling) raise SPAWN[1]; if underground lower it and FALL_Y_THRESHOLD.

**Verify:** env server accepts `--env hunting_wild` without error; first episode places agent on terrain; pig spawns within the bedrock cage.

---

## 6. Caveats

- **Reward or obs-layout changes invalidate policy checkpoints.**
- **Ensemble/probabilistic checkpoints** will not load into the old single deterministic model, and vice-versa (expected; these are new experiment runs, not a migration).
- **Aim shaping + auto-fire (Phase 1)** fix the hard-exploration bottleneck. Without them, `attack` requires `los_hit` (ray exactly on pig) which random exploration almost never achieves.
- **Demo quality matters**: scripted demos should include aligned + in-range attack sequences. `generate_hunting_demos.py` produces this automatically; human demos use X/F for attack.
- **BC anchor decays to zero by end of training** — intentional. Early anchor prevents imagination actor drift; decay lets the agent eventually surpass the demonstrator.
- **`hunting_wild` spawn calibration**: seed 4837 targets y=65 for plains but terrain height varies ±3 blocks. On first run check `YPos` in episode logs and adjust `SPAWN[1]` and `FALL_Y_THRESHOLD` in `hunting_wild_cfg.py` if the agent falls.
- **`env_server` exits on client disconnect** — restart it before each fresh run.
- Gating/intrinsic require an **ensemble** (`ENSEMBLE_SIZE>1`); with a single model they are inert and the extra log columns report zeros.

---

## 7. Video-based world model (pixel-space RSSM) — a SEPARATE architecture (2026-08-01)

**Why:** everything above predicts directly in Malmo's privileged, hand-engineered observation space (exact entity coordinates, a voxel occupancy grid) — state that a real robot never gets. Frontier robotics world models (DreamerV1/V3 on real robots, e.g. DayDreamer; Genie/Cosmos/GAIA at larger scale) learn from pixels instead. This phase adds a pixel-space world model as a second, independent architecture so the project also demonstrates that direction, without touching or risking the validated vector pipeline above.

**Scope decision (v1): single RSSM, no ensemble.** The vector side's aleatoric/epistemic uncertainty split is NOT replicated here yet — the RSSM's stochastic latent already captures aleatoric spread on its own, and an ensemble-of-RSSMs would be ~5x the world-model compute for a first pass. Parity with the vector-side uncertainty story is explicit future work (see below), not a v1 goal.

### 7.1 Architecture
A DreamerV1-style RSSM, fully separate from `models/world_model.py`:
```
frame (64x64x3) + hybrid state vector (98-dim)
  -> ConvEncoder + vec MLP -> embed
  -> RSSM (GRU deter=256 + Gaussian stoch=32, KL vs free-nats floor)
  -> feat = [stoch ++ deter] (288-dim)
  -> ConvDecoder (frame recon) + vec head + reward/cont heads (conditioned on feat+action)
```
Imagination is prior-only (`img_step`), rolled from real posterior start states — same Dreamer principle as the vector agent, in pixel space. ~5.5M params.

**New files:**
- `models/video_world_model.py` — `ConvEncoder`, `ConvDecoder`, `RSSM`, `VideoWorldModel` (the RSSM + all heads + `loss()`)
- `models/latent_actor_critic.py` — policy/value on RSSM features; **NOT checkpoint-compatible** with `ActorCritic` (different input space)
- `algos/sequence_replay_buffer.py` — episode-aware buffer storing frames as uint8 (not float32), with padding + masking so short (e.g. fast-kill) episodes aren't discarded
- `algos/dreamer_video.py` — `DreamerVideo(BaseAgent)`, a **separate class from `Dreamer`, not a subclass** (stateful per-env RSSM tracking, sequence collection, and prior-only imagination substrate all differ enough that sharing code would force branches into the frozen vector path)
- `training/configs/hunting_video_cfg.py` — claims the `WM_LATENT_DIM` / `USE_GRU` / `WM_KL_SCALE` / `WM_FREE_NATS` / `WM_SEQ_LEN` seams that `world_model_cfg.py` reserved back in Phase 2 (§2.2) and that no code had read until now
- `visualization/dream_strip_viz.py` — film-strip figure: real context frames / posterior reconstructions / imagined continuation, with predicted reward+continuation per column

**Gated additions to existing files** (every existing algo/env combination is byte-identical — verified via empty `git diff` on `dreamer.py`, `world_model.py`, `replay_buffer.py`, `actor_critic.py`, `hunting_cfg.py`, `imagination_viz.py`, and the mission XML):
- `world_model_cfg.py` — additive `VIDEO_ENABLED=False` / RSSM-size / loss-weight block
- `envs/hunting_env.py` — `mission.requestVideo(W,H)` + frame harvesting in `_get_obs_dict`, gated on `cfg.VIDEO_ENABLED`
- `envs/env_server.py` — `attach_frame()` base64-encodes `env.last_frame` onto the reset/step JSON payload only when the env is video-enabled; registers `hunting_video`
- `envs/env_client.py` — `EnvClient(..., video=False)`; when `True`, `reset()`/`step()` return `(vec, frame)` tuples instead of a flat vector
- `training/train.py` — `dreamer_video`/`hunting_video` registry entries, `LatentActorCritic` vs `ActorCritic` selection, list-based (not ndarray) obs container for video, and a hard error if `--algo dreamer_video` is paired with a non-video env (or a warning the other way — frames would ship over the wire and be discarded)

### 7.2 Verification done (no Malmo, `tests/smoke_video_wm.py` / `smoke_dreamer_video.py` / `smoke_wire.py`)
Encoder/decoder shape round-trips; `VideoWorldModel.loss()` finite with gradients reaching every parameter; KL free-nats floor behaves correctly on identical prior/posterior; `SequenceReplayBuffer` padding/masking/dtype-preservation, including a real bug found and fixed (a single episode longer than buffer capacity was being evicted to empty — fixed by never evicting the last remaining sealed episode); full collect→update loop against a fake video env with all losses finite; **learnability sanity check** (`vwm_image` measurably decreases fitting a constant-frame env); per-env RSSM state correctly resets to initial on episode `done`; checkpoint save→load round-trip reproduces identical action logits; wire-format base64 round-trip and cross-compatibility in both directions (video client vs non-video server and vice versa); Python-3.6 `py_compile` gate on every env-server-side file; the project's own `utils/validate.py` Tier-1 checks (registry parity across `env_server.py`/`train.py`, 73 files of valid syntax); `dream_strip_viz.py` end-to-end render against a checkpoint (both pure-imagination and real-context modes).

### 7.3 What's NOT done — needs live Malmo (cannot be verified offline)
1. Frame arrival sanity check against real Minecraft (non-black frames, correct 64x64 resolution, `requestVideo` accepted by the running Malmo build)
2. Frame/observation sync check (yaw vs. rendered frame content, quantify the skew — `_get_obs_dict` harvests frames opportunistically per poll, not on a guaranteed same-tick basis)
3. Confirm `--env hunting --algo dreamer` (the existing vector path) shows zero wire-format or step-timing regression
4. First real training run: `python training/train.py --env hunting_video --algo dreamer_video --base-port 9999`, and a first look at whether `vwm_image`/`vwm_kl` trend down against real footage

### 7.4 Explicit future work (not v1)
Ensemble-of-RSSMs for the aleatoric/epistemic split (parity with §2.2's uncertainty story, in pixel space); DreamerV3-style categorical latents + symlog transforms; DreamerFD demo-sequence integration for the video path (currently vector-only, §5.2); binary/JPEG frame transport (currently base64-in-JSON, ~25KB/step — fine on localhost, not optimized); latent-policy support in `evaluate.py`/`live_viz.py` (they currently only know how to build `ActorCritic`).
