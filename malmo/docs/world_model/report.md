# World Models for MalmoRL — Uncertainty-Aware Imagination (Hunting)

*Status: 2026-07-02 · branch `extension/world-model-hunting` · built + validated offline, ready for Phase D (live Malmo).*

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

## 5. Caveats

- **Reward or obs-layout changes invalidate policy checkpoints.**
- **Ensemble/probabilistic checkpoints are a new architecture** — they will not load into the old single deterministic model, and vice-versa (expected; these are new experiment runs, not a migration).
- **Aiming is the hard part of the task.** `los_hit` requires the crosshair on the pig before an `attack` lands; dense `approach`+`aim`+`hit` shaping and `penned` mode exist to bootstrap this. If cold-start exploration stalls, widen `ATTACK_RANGE` or start the pig closer before touching the world-model knobs.
- **`env_server` exits on client disconnect** — restart it before each fresh run.
- Gating/intrinsic require an **ensemble** (`ENSEMBLE_SIZE>1`); with a single model they are inert and the extra log columns report zeros.
