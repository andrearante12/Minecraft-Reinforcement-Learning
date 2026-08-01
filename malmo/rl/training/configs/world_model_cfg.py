"""
training/configs/world_model_cfg.py
-----------------------------------
Hyperparameters for the model-based RL (world-model / Dreamer) agent.

This is a *mixin*: BaseCFG inherits from it so every environment config
gains sane world-model defaults without affecting model-free runs
(PPO/DQN/BC never read these fields).

The world model is a compact dynamics model that predicts the next
observation, reward, and episode-continuation from (obs, action). Because
MalmoRL observations are already low-dimensional state vectors, the model
predicts directly in *observation space* (proprio/goal as residual deltas,
voxel grid as binary occupancy) — no image VAE needed.

The Dreamer agent then trains the actor-critic almost entirely inside this
model's "imagination", drastically cutting the number of expensive real
Malmo steps.
"""


class WorldModelCFG:
    # ── World-model network (multi-stream encoder mirrors ActorCritic) ──────────
    WM_PROPRIO_HIDDEN = 64
    WM_GOAL_HIDDEN    = 64
    WM_VOXEL_HIDDEN   = 128
    WM_ACTION_EMBED   = 16
    WM_HIDDEN         = 256    # shared trunk / latent feature width
    WM_HEAD_HIDDEN    = 128    # reward & continuation head width

    # ── World-model loss weights ────────────────────────────────────────────────
    WM_W_PROPRIO = 1.0
    WM_W_GOAL    = 1.0
    WM_W_VOXEL   = 1.0    # binary cross-entropy on the occupancy grid
    WM_W_REWARD  = 1.0
    WM_W_CONT    = 1.0    # binary cross-entropy on P(not done)

    # ── World-model optimisation ────────────────────────────────────────────────
    WM_LR              = 3e-4
    WM_BATCH_SIZE      = 256
    WM_GRAD_STEPS      = 1        # world-model SGD steps per update() call
    WM_BUFFER_CAPACITY = 100000
    WM_MAX_GRAD_NORM   = 100.0

    # ── Imagination (behaviour learning) ────────────────────────────────────────
    IMAG_HORIZON      = 15       # H: how many steps to dream ahead
    IMAG_BATCH        = 256      # number of start states per imagined update
    IMAG_LAMBDA       = 0.95     # λ for Dreamer λ-returns
    IMAG_UPDATES      = 1        # actor-critic updates per update() call
    IMAG_ENTROPY_COEF = 0.003    # entropy bonus for the actor in imagination
    IMAG_VALUE_COEF   = 0.5      # critic-loss weight

    # ── Collection schedule ─────────────────────────────────────────────────────
    LEARNING_STARTS  = 1000      # real transitions before any learning
    REAL_TRAIN_EVERY = 1         # run update() every N real env steps (post-warmup)

    # ── Stochastic latent / RSSM variant (Phase 2 — default OFF) ────────────────
    # When WM_LATENT_DIM > 0 / USE_GRU True, the world model carries a recurrent
    # stochastic latent so it can represent the un-actioned (e.g. a fleeing
    # animal's) part of the dynamics. Not used by the deterministic Phase-0 model.
    WM_LATENT_DIM = 0
    USE_GRU       = False
    WM_KL_SCALE   = 1.0
    WM_FREE_NATS  = 1.0
    WM_SEQ_LEN    = 1

    # ── Uncertainty-aware imagination (default OFF — enabled per-env) ───────────
    # An ensemble of probabilistic world models lets the agent measure and split
    # its predictive uncertainty on the moving target into aleatoric (irreducible
    # noise) and epistemic (model ignorance), then act on each differently:
    #   • horizon gating  — shorten the effective imagination horizon where the
    #     future is genuinely unpredictable (trust factor folded into the
    #     GAMMA*continuation discount), so value targets aren't poisoned by
    #     diverged dreams.
    #   • exploration     — reward the actor for visiting states the model is
    #     merely IGNORANT about (epistemic), never merely NOISY (aleatoric) —
    #     the guard against the "noisy-TV" trap.
    # Defaults here keep every existing PPO/DQN and deterministic-Dreamer run
    # byte-identical; the hunting config turns these on for the study.
    ENSEMBLE_SIZE   = 1        # K; >1 → aleatoric/epistemic decomposition
    WM_PROBABILISTIC = False   # goal head emits (mean, log_var) → Gaussian NLL
    WM_MIN_LOGVAR   = -8.0     # clamp for NLL stability
    WM_MAX_LOGVAR   =  2.0
    WM_INIT_LOGVAR  = -2.0     # untrained log-var bias (modest cold-start variance)
    WM_UNCERTAINTY_DIMS = (0, 1, 2)  # GOAL-stream dims carrying target position

    WM_HORIZON_GATING = False  # τ_t = exp(-WM_TRUST_BETA * cumulative_uncertainty)
    WM_TRUST_BETA     = 0.5

    INTRINSIC_COEF = 0.0       # weight on the disagreement intrinsic reward
    INTRINSIC_KIND = "epistemic"  # epistemic | total  (never aleatoric)

    # ── DreamerFD-style demo integration (default OFF — supply --demo-path) ────
    # A separate demo replay buffer feeds WM training (DEMO_WM_FRACTION of each
    # minibatch) and imagination start states (DEMO_START_FRACTION of each batch).
    # A decayed BC cross-entropy anchor discourages the imagination actor from
    # drifting far from the expert early in training.
    # Set DEMO_PATH via --demo-path CLI flag; the others are tuning knobs.
    DEMO_PATH           = None   # path to demo JSON; None → demos disabled
    DEMO_WM_FRACTION    = 0.1    # fraction of each WM minibatch drawn from demos
    DEMO_START_FRACTION = 0.3    # fraction of imagination starts drawn from demos
    DEMO_BC_COEF        = 0.0    # BC loss coefficient (0 = off)

    # ── Video world model (RSSM; used ONLY by --algo dreamer_video) ─────────────
    # A separate architecture from the vector WorldModel above: a DreamerV1-style
    # RSSM (conv encoder -> GRU deterministic state + Gaussian stochastic latent
    # -> conv decoder), trained on frame SEQUENCES from a video-enabled env
    # (VIDEO_ENABLED=True, e.g. hunting_video). Off by default; every existing
    # algo/env combination is unaffected. The WM_LATENT_DIM / USE_GRU /
    # WM_KL_SCALE / WM_FREE_NATS / WM_SEQ_LEN seams above are claimed by this
    # variant (models/video_world_model.py) — they are read by no other code.
    VIDEO_ENABLED  = False   # env_server attaches a frame; EnvClient returns (vec, frame)
    VIDEO_WIDTH    = 64
    VIDEO_HEIGHT   = 64
    VIDEO_CHANNELS = 3

    # RSSM sizes
    RSSM_DETER        = 256   # GRU deterministic state size
    RSSM_HIDDEN       = 256   # width of prior/posterior/GRU-input MLPs
    RSSM_EMBED        = 1024  # CNN embedding size (fixed by the 64x64 conv stack)
    VWM_PROPRIO_EMBED = 128   # proprio-vector embedding fed to the posterior
    VWM_CNN_DEPTH      = 32   # base channel count (32 -> 64 -> 128 -> 256)

    # Optimisation / loss weights
    VWM_LR        = 3e-4
    VWM_W_IMAGE   = 1.0
    VWM_W_PROPRIO = 1.0   # vector reconstruction head (hybrid obs)
    VWM_W_REWARD  = 1.0
    VWM_W_CONT    = 1.0

    # Sequence replay
    VWM_SEQ_BATCH       = 16      # sequences per world-model minibatch
    VWM_MIN_SEQ         = 8       # shortest usable episode fragment (padded+masked to WM_SEQ_LEN)
    SEQ_BUFFER_CAPACITY = 50000   # steps of uint8 frames stored (~600MB at 64x64x3)
