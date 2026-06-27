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

    # ── Research seams (default OFF — reserved follow-ups) ──────────────────────
    ENSEMBLE_SIZE  = 1     # >1 → ensemble for disagreement-based exploration
    INTRINSIC_COEF = 0.0   # weight on ensemble-disagreement intrinsic reward
