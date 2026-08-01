"""
training/configs/hunting_video_cfg.py
--------------------------------------
Video-observation variant of the hunting task config, for the RSSM-based
`dreamer_video` algo (models/video_world_model.py, algos/dreamer_video.py).

Runs on BOTH conda envs (malmo Py3.6 env_server, train_env training) — keep
this file Python-3.6-safe (no f-strings, no dataclasses).

Same mission/env/rewards as HuntingCFG; only the video + RSSM knobs differ.
"""

from training.configs.hunting_cfg import HuntingCFG


class HuntingVideoCFG(HuntingCFG):
    # ── Video observation ────────────────────────────────────────────────────
    VIDEO_ENABLED = True

    # ── Claim the reserved RSSM seams (see world_model_cfg.py) ──────────────
    WM_LATENT_DIM = 32      # stochastic latent size
    USE_GRU       = True
    WM_KL_SCALE   = 1.0
    WM_FREE_NATS  = 1.0
    WM_SEQ_LEN    = 16      # training sequence length

    # ── v1 scope: single RSSM, no ensemble/uncertainty machinery ────────────
    # (the vector-obs hunting_cfg.py enables these for the aleatoric/epistemic
    # study; the video path starts single-model — see docs/world_model/report.md)
    ENSEMBLE_SIZE     = 1
    WM_PROBABILISTIC  = False
    WM_HORIZON_GATING = False
    INTRINSIC_COEF    = 0.0

    SAVE_EVERY = 25
