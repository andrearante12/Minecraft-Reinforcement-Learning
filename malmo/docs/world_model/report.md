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

## 8. Live split-screen viewer (2026-10-09)

One browser tab with the Minecraft camera on the left and the world model's predictions for the **same step** on the right. Camera frames ride the existing env socket, one per step (~6.6 fps), so each frame is exactly the observation the predictions were computed from.

| Model | Right half |
|---|---|
| `dreamer` (vector ensemble) | **top-down map** (north up, live): real agent/pig + trails, each of the K members' imagined agent and pig paths under ONE shared action plan (their spread = epistemic), 1σ discs from each member's predicted pig-Δ variance (aleatoric); then aleatoric/epistemic gauges (raw value printed; the gauge reads x/(x+EMA x)), obs readout, net activations, π(a\|s), V(s) |
| `dreamer_video` (pixel RSSM) | real 64×64 input next to the posterior decode ("model sees"), an 8-step prior-only dream strip with r̂/ĉ, π(a\|s), V(s) |

The layout is picked automatically from the agent/checkpoint type.

The right half has two tabs, and it opens on the one that matches the source:
- **Inference:** what the model predicts right now. The top-down map (heading-up, so it matches the camera), π(a|s), V(s) and the network.
- **Training:** whether it's learning. Tiles show episodes, kill rate and mean return over the last 20 episodes, env steps, updates and replay-buffer size. Below them are the aleatoric/epistemic cards and charts.
  - Charts: episode return, uncertainty over time, effective imagination horizon, world-model loss, imagined return and policy entropy. For `dreamer_video`, RSSM loss and KL replace the vector-only charts.
  - Loss curves only exist in a training run (`train.py --viz-port`). Checkpoint playback shows returns and per-step uncertainty.
  - The server keeps the stats history, so a page opened mid-run gets the full curves.

```bash
# T1 — Minecraft, visible or headless
cd malmo/Minecraft && xvfb-run -a -s "-screen 0 1280x720x24" ./launchClient.sh

# T2 — env server. The vector env needs MALMO_RECORD_VIDEO=1 to ship camera frames (640x360;
#      MALMO_VIDEO_W/H to change). hunting_video already ships its 64x64 frames:
#      do NOT set MALMO_RECORD_VIDEO there (it would override the model's input size).
MALMO_RECORD_VIDEO=1 python malmo/rl/envs/env_server.py --env hunting --port 9999

# T3a — watch while it trains (pauses with a banner during agent.update())
python malmo/rl/training/train.py --env hunting --algo dreamer --viz-port 8765
# T3b — or play back a checkpoint (algo auto-detected from the checkpoint keys)
python malmo/rl/visualization/live_split.py --env hunting --checkpoint malmo/rl/checkpoints/<ckpt>.pt
# → open http://localhost:8765/
```

- **Third person:** add `MALMO_VIDEO_VIEWPOINT=1` to the env_server command (`2` = facing the agent). It sets the viewpoint on Malmo's video stream, which ignores in-game F5. Don't use it with `hunting_video`, since it changes the pixel model's input.
- **No hand/sword in first person?** `malmo/Minecraft/run/options.txt` has `hideGUI:true` saved (left by an F1 press), and that hides the first-person hand. Set it to `false` with the client closed. In third person the sword shows either way.
- `--viz-env k` streams parallel env k; `--viz-third-person` / `--third-person` press F5+F1 in the client (`utils/send_mc_keys`; needs the right `DISPLAY` under Xvfb).
- No browser connected → no frames are built (zero training overhead). The server is stdlib-only (`visualization/live_server.py`, SSE); train_env needs no new packages.
- No Malmo: `live_split.py --mock` (synthetic obs + camera; `--checkpoint` optional). Offline test: `python malmo/rl/tests/smoke_live_split.py`.
- Files: `visualization/live_server.py`, `visualization/live_frames.py`, `visualization/live_split.py`, `site/live.html`, `site/js/live.js`.

## 9. Vision-first rabbit hunting: frozen DINOv2 + seeded real world (`hunting_vision`, 2026-10-09)

**Why:** the vector hunting task never needed a world model. Its `fleeing` mode only spawned the pig farther away (the pig never fled), the arena was an open 18×18 box, and the agent was given the pig's exact coordinates. That is reactive pursuit, which model-free RL can handle. `hunting_vision` changes three things:
- **Prey that flees.** Rabbits run from the player on their own, so the agent has to predict where they'll go and intercept them.
- **A real seeded Minecraft world.** Trees and hills block the view, so the RSSM's recurrent state has to remember where the rabbit went.
- **Pixels + proprioception only.** The camera goes through a **frozen pretrained DINOv2 ViT-S/14**, and the agent gets 11 proprio dims (no target coordinates). That's what a robot honestly has.

The env still knows the rabbit's position, uses it for reward, and sends it out as `info["privileged"]` for the scripted teacher and the feature probe. The model never sees it.

### 9.1 Curriculum rungs (`HUNT_LEVEL`; env server: `MALMO_HUNT_LEVEL=<n>`)
| Rung | World | Prey | AUTO_ATTACK | Purpose |
|---|---|---|---|---|
| L0 | flat walled arena (`hunting_vision_flat.xml`) | 1 rabbit | on | fast-reset debug rung: first kill, pipeline bugs |
| **L1** | seed 4837, 42×42 bedrock cage | 1 rabbit | on | main task: occlusion + fleeing |
| L2 | same | 1 rabbit | off | the agent aims from pixels |
| L3 | 64×64 cage | 3 rabbits | off | longer horizon, target choice |

The rung never changes observation or action shapes, so only the env server needs to know it. The cage and rabbits are injected at reset (`HuntingEnv._build_mission_xml`), and the seed is substituted from `WORLD_SEED`.

The rabbit has 3 HP and the wooden sword deals 4, so the first hit kills. A rabbit can vanish before life 0 is ever observed, so kills are detected from Malmo's own damage event (`RewardForDamagingEntity`, `USE_DAMAGE_SIGNAL`) plus a drop in the rabbit count. The action set is hunting's 11 actions plus `jump_forward`, because natural terrain has 1-block steps.

### 9.2 Vision pipeline
```
Malmo frame 224x224 ──(env server, base64)──> trainer
  └─ FrozenEncoder (DINOv2 ViT-S/14, fp16, ~5 ms/frame on the 4070)
       CLS + 16x16 patches avg-pooled to 4x4  ->  tokens (17, 384) fp16
  └─ replay stores tokens (+ 64x64 thumbnail for the viewers), never raw pixels
tokens + proprio(11) -> FeatureEncoder (trainable) -> RSSM -> FeatureDecoder (reconstruct tokens, DINO-WM style)
                                                         └-> thumbnail ConvDecoder on a STOP-GRAD latent (viz only)
```
- The backbone is never trained, never checkpointed, and never enters an optimizer. The checkpoint records `[VIS_ENCODER, VIS_POOL_GRID]` and refuses to load under a different encoder.
- `VWM_INPUT="pixels"` (the default) builds exactly the old `hunting_video` modules.

### 9.3 Demos: frames + actions (`utils/demo_io.py`)
- **Format.** A demo set is a directory: `index.json` plus one `ep_NNNN.npz` per episode. Each episode stores JPEG frames, proprio, actions, rewards, dones, and the privileged target stream. Encoder tokens are cached per encoder next to each episode.
- **Privileged teacher.** `utils/generate_hunting_demos.py --env hunting_vision` acts on the rabbit's true position: lead pursuit, pitch onto a low target, and a jump when stuck. The saved demos hold what the student sees (learning by cheating).
- **Human demos.** `utils/record_demos.py --env hunting_vision` records human play; Space is `jump_forward`.
- **DreamerFD in DreamerVideo.** With `--demo-path <dir>`:
  - `DEMO_WM_FRACTION` of each world-model batch comes from demos;
  - `DEMO_START_FRACTION` of imagination starts come from demo posteriors;
  - a BC anchor (`DEMO_BC_COEF`) decays to 0 over training;
  - `--pretrain-wm-steps N` fits the world model on demos before the first live step.

### 9.4 Runbook
```bash
# T1 — Minecraft
cd malmo/Minecraft && ./launchClient.sh
# T2 — env server (conda: malmo). Do NOT set MALMO_RECORD_VIDEO (it would override the 224x224 input).
MALMO_HUNT_LEVEL=0 python malmo/rl/envs/env_server.py --env hunting_vision --port 9999
# T3 (conda: train_env)
#   a) scripted demos for this rung      -> malmo/demos/hunting_vision_L0/
python malmo/rl/utils/generate_hunting_demos.py --env hunting_vision --port 9999 --episodes 50
#   b) can DINOv2 see the rabbit?  (offline — reads the demo directory only)
python malmo/rl/visualization/feature_probe.py --demos malmo/demos/hunting_vision_L0
#   c) train  (restart T2 first: env_server exits when a client disconnects)
python malmo/rl/training/train.py --env hunting_vision --algo dreamer_video \
    --demo-path malmo/demos/hunting_vision_L0 --pretrain-wm-steps 2000 --viz-port 8765
```
Set the same `MALMO_HUNT_LEVEL` for the demo generator and recorder in T3: they read it for the teacher's `AUTO_ATTACK`/`MAX_STEPS` and for the default output folder.

### 9.5 Verified offline
- **`tests/smoke_hunting_vision_env.py`** (malmo env, Py3.6):
  - every rung's XML passes Malmo's schema check;
  - plain `hunting` XML is byte-identical to the old construction;
  - levels don't leak into each other;
  - observations are 11-dim with the privileged stream still computed;
  - damage-signal kill detection works, and a rabbit disappearing without damage is not a kill.
- **`tests/smoke_dreamer_vision.py [--real]`** (train_env; stub and real DINOv2):
  - encoder frozen, fp16, (17, 384) tokens;
  - buffer keeps dtypes;
  - feature world model loss is finite, and the stop-grad thumbnail decoder sends zero gradient into the RSSM;
  - feature loss falls on a fixed batch;
  - `demo_io` round-trip, append and feature cache work;
  - DreamerVideo plus demos: pretraining, 90+ finite updates, a constant log-key set, demo rows in the world-model batch, BC decay;
  - checkpoint round-trip works, and a mismatched encoder is refused.
- **Regressions:** the existing `smoke_video_wm`, `smoke_dreamer_video`, `smoke_wire` and `smoke_live_split` all pass. Tier-1 `validate.py` passes, with registry parity at 19 envs.

### 9.6 Needs live Malmo (Phase A, in order)
1. Frames arrive at 224×224. Seed 4837's caged region is usable: grass, trees, no water or cliffs. Calibrate `SPAWN[1]`, `TARGET_SPAWN_Y` and the XML `Placement` y.
2. Rabbits flee and a sprinting agent can catch them. The first hit kills, and `killed` fires.
3. Measure reset time. `forceReset="true"` regenerates the world each episode. If that's too slow, keep the world loaded and clean up leftover rabbits and drops with `ChatCommands` instead.
4. Feature probe. If the bearing R² from the 4×4 grid is close to the CLS-only number, set `VIS_POOL_GRID = 8`.
5. L0 training with demos until the first `killed`. Then L1.

**Then Phase D:** the same live-step budget for `dreamer_video`+features vs PPO on the same frozen features (needs a feature-input `ActorCritic`, not built yet) vs vector `dreamer` (privileged upper bound), across L0–L3.

### 9.7 First live run and fixes (2026-10-10)

**Rabbit L0 run: 27 episodes, 0 kills.**
- The policy collapsed to `sprint_forward`, which went from about 10% of actions to 75–90%. Entropy fell from 2.48 to about 0.8, and the agent spent 50–74% of its time in corners.
- Imagined return was about **+5** while real returns were −13 to −22. The policy was exploiting a wrong world model.

**Root cause: confounded demos.** The teacher only sprints once it's aimed, so in the demos `sprint_forward` averaged +0.23 reward and turns averaged −0.2. The agent's own sprints averaged −0.08. The latent was trained only on whole-frame DINOv2 reconstruction, which is dominated by grass, walls and sky, so it didn't encode where the small rabbit was. The reward model therefore learned "sprint = reward" unconditionally.

**Fixes:**
- **`VWM_PRIV_AUX`.** A training-time head predicts the target's bearing (sin and cos), distance and in-range flag from both the posterior and the prior latent. The labels come from `info["privileged"]`, stored as buffer key `priv`. Inputs are unchanged (pixels + proprio), but the latent now has to track the prey, including inside imagination.
- **`VWM_SYMLOG_REWARD`.** The reward head predicts symlog(r), as in DreamerV3, so the rare kill reward no longer spikes the loss.
- **`DEMO_ONLY_SUCCESS`.** Failed teacher chases are dropped from the demos.
- **`DEMO_BC_COEF` 0.5 → 0.2.**
- **Prey is now a cow** (`PREY`, `MALMO_HUNT_PREY`). It is large and easy to see, wanders, and panics after each hit; three wooden-sword hits kill it. Rabbits remain available as the harder target. Demo folders are now named `demos/hunting_vision_<prey>_L<level>/`.
- **`<MissionQuitCommands/>` added to both vision missions.** Without it the env's `quit` was ignored, and missions only ended on the server timer.

### 9.8 Cow L0 result and the move to real terrain (2026-10-10)

**Cow L0 result:** 1,266 episodes, run overnight.

| Episodes | Kill rate |
|---|---|
| 1–25 | 24% |
| 201–300 | 98% |
| 300–1266 | 88–98% (per-100 window) |

Mean kill length fell from 130 to about 100 steps. The final checkpoint is pinned as `checkpoints/cow_L0_final_ep1266.pt`, and `ep500` is kept as the first window where every episode was a kill. `train.py` now prunes old checkpoints (`KEEP_CHECKPOINTS = 3` + `KEEP_CHECKPOINT_EPISODES`), because 116 MB every 25 episodes filled the disk overnight.

**Seed 4837 was unusable.** Its origin is jungle, and the agent spawned underground at y 65 (the surface is about 67). The cow died to the world there and that counted as a kill. Two fixes:
- With `USE_DAMAGE_SIGNAL`, kills and hit rewards now require damage dealt by the agent.
- **`utils/scout_seeds.py`** (malmo env) loads each candidate seed with a creative agent high above the origin. It reads a 43×41×43 block grid and scores water %, tree-canopy %, median surface height and roughness. With `--save-heightmap` it writes `envs/hunting_vision/heightmaps/<seed>.json`.

With `cfg.HEIGHTMAP` set, `HuntingEnv` places things on the ground instead of at a fixed y:
- the agent spawns on the nearest standable column to the origin;
- prey spawn only on standable ground, never on a treetop, at the exact surface height;
- `Placement`, `SPAWN` and `FALL_Y_THRESHOLD` are all derived from the heightmap.

**New levels:**

| Level | World | Auto-attack | Max steps |
|---|---|---|---|
| L1 | plains seed **85253262** (0% water, 0% trees, roughness 1.8) | on | 600 |
| L2 | forest seed **104387003** (0% water, 46% canopy, roughness 3.2) | on | 600 |
| L3 | forest | off | 600 |
| L4 | forest, 3 cows | off | 900 |

The step limits were raised because L0's 200 steps caused most of its timeouts.

**Live L1 check:** the agent and cow both stand on the ground and the cow is sensed. Resets take 10–40 seconds, because `forceReset` regenerates the world each episode. The teacher kills 4 of 4 on plains.
