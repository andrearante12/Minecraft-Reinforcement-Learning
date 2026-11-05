"""
training/configs/hunting_vision_cfg.py
---------------------------------------
Vision-first rabbit hunting in a real seeded Minecraft world, for
`--algo dreamer_video` with a FROZEN pretrained visual encoder (DINOv2).

What the agent observes (no privileged target info):
  - camera frame (224x224) -> frozen DINOv2 ViT-S/14 -> CLS + GxG pooled patch
    tokens (encoded in the TRAINER process; the env server never touches torch)
  - 11-dim proprioception (onGround, yaw, pitch, velocity, pos-rel-spawn, food,
    health) — what a robot honestly knows about itself

The env still computes the rabbit's position internally for the REWARD and
ships it to scripted teachers as info["privileged"]; it never reaches the model.

Why rabbits in a natural world: rabbits flee the player on their own (real,
stochastic, agent-reactive dynamics → the agent must predict and intercept),
and trees/hills occlude them (→ the RSSM's recurrent state must remember where
the prey went). Both are what a world model is for.

Curriculum rungs (HUNT_LEVEL, overridable in the env-server process with
MALMO_HUNT_LEVEL=<n>; obs/action shapes never change across rungs, so the
trainer does not need to know the rung):
  L0 debug — flat walled arena, auto-attack on. Fast resets; a smoke rung for
             the first kill / pipeline bugs, not a research result.
  L1       — seeded flat MEADOW with tree clumps (trees, not hills, hide the
             prey), fenced 42x42 pasture, 3 cows; the episode ends when all 3
             are dead (each kill pays) or after 900 steps; auto-attack on.
  L2       — same, auto-attack off: the agent aims from pixels itself.

Runs on BOTH conda envs (malmo Py3.6 env_server, train_env training) — keep
this file Python-3.6-safe (no f-strings, no dataclasses).
"""

import os
from training.configs.hunting_cfg import HuntingCFG

_MISSIONS = os.path.join(HuntingCFG.ROOT_DIR, "envs", "hunting_vision", "missions")
_HEIGHTMAPS = os.path.join(HuntingCFG.ROOT_DIR, "envs", "hunting_vision", "heightmaps")

# Seeds picked with utils/scout_seeds.py (43x43 around the origin); their saved
# heightmaps put the agent and prey exactly on the ground:
#   523467673 meadow — flat grass (roughness 0.89) with tree clumps (46% canopy),
#             one small pond + a giant mushroom: obstruction without hills. USED.
#   85253262  plains — 0% trees but hilly (roughness 1.8): hills hid the cow.
#   104387003 forest — 46% canopy, hilly (roughness 3.2).
#   229543111 tundra — flat, open snow (roughness 0.98).
MEADOW_SEED = 523467673
PLAINS_SEED = 85253262
FOREST_SEED = 104387003
TUNDRA_SEED = 229543111
DESERT_SEED = 332745816

BIOME_SEEDS = {"meadow": MEADOW_SEED, "plains": PLAINS_SEED, "forest": FOREST_SEED,
               "tundra": TUNDRA_SEED, "desert": DESERT_SEED}


class HuntingVisionCFG(HuntingCFG):
    # ── Mission / world (defaults = L1; see LEVELS) ──────────────────────────
    MISSION_FILE     = os.path.join(_MISSIONS, "hunting_vision.xml")
    WORLD_SEED       = MEADOW_SEED   # substituted into DefaultWorldGenerator
    HEIGHTMAP        = os.path.join(_HEIGHTMAPS, "{0}.json".format(MEADOW_SEED))
    SPAWN            = (0.5, 71.0, 0.5)   # replaced from HEIGHTMAP by HuntingEnv
    FALL_Y_THRESHOLD = 60.0               # replaced from HEIGHTMAP by HuntingEnv
    CAGE_HALF        = 21            # boundary drawn at reset (None = none)
    CAGE_STYLE       = "barrier"     # invisible barrier blocks (nothing to see, nothing crosses)
    CAGE_FLOOR_Y     = 61
    CAGE_TOP_Y       = 80
    MAX_STEPS        = 900           # 3 kills per episode; L0's 200 often timed out mid-chase
    # The step limit, not wall-clock, must end episodes: with 3 clients in lockstep
    # a step takes ~0.55 s (PPO) to ~0.7 s (Dreamer, updating every step), so the
    # base 150 s mission timer cut episodes at ~200-270 steps, and at different
    # lengths for the two agents. 20 min covers 900 steps at up to ~1.3 s/step.
    MISSION_TIME_LIMIT_MS = 1200000

    # ── Prey (PREY picks a row of PREYS; env var MALMO_HUNT_PREY overrides) ───
    # cow    — large, easy to see; wanders, panics + sprints off when hit; 10 HP =
    #          3 wooden-sword hits, so a kill is hit -> chase -> hit. Start here.
    # rabbit — small (hard for the frozen encoder at range), flees the player on
    #          its own, one-hit kill. The harder target once cows work.
    PREY  = "cow"
    PREYS = {
        "cow":    {"TARGET_MOB": "Cow",    "TARGET_MAX_LIFE": 10.0, "TARGET_HEIGHT": 1.4},
        "rabbit": {"TARGET_MOB": "Rabbit", "TARGET_MAX_LIFE": 3.0,  "TARGET_HEIGHT": 0.5},
    }
    TARGET_MOB       = "Cow"
    TARGET_MAX_LIFE  = 10.0
    TARGET_HEIGHT    = 1.4           # body height: the teacher aims at its middle
    # Other animals that also count as prey (targeting, hits, kills). The worlds
    # have natural pigs; hitting or killing one pays the same as a cow. Same 10 HP.
    # Per-mob body heights for aiming; the spawned TARGET_MOB uses TARGET_HEIGHT.
    EXTRA_PREY_MOBS  = ("Pig",)
    # Kill oracle: a kill = the player's MobsKilled stat rose AND a prey animal
    # (by entity id) within ATTACK_RANGE + KILL_REACH_MARGIN last step is gone.
    # No false kills from hit animals wandering out of range or despawning.
    KILL_ORACLE       = True
    KILL_REACH_MARGIN = 2.0
    PREY_HEIGHTS     = {"Cow": 1.4, "Pig": 0.9, "Rabbit": 0.5}
    TARGET_SPAWN_Y   = 71.0          # only used without a HEIGHTMAP
    N_TARGETS        = 3             # several cows: more encounters per (slow) world reset
    KILLS_TO_END     = 3             # each kill pays REWARD_KILL; episode ends when all are dead
    TARGET_MODE      = "fleeing"     # = spawn at least FLEEING_MIN_DIST away
    ARENA_MIN        = -18.0
    ARENA_MAX        = 18.0
    FLEEING_MIN_DIST = 8.0
    DIST_SCALE       = 30.0
    AUTO_ATTACK      = True
    # Train on 4 biomes (equal weight); FOREST is HELD OUT for evaluation only.
    # Evaluate / override in the env-server process: MALMO_WORLD_SEEDS=forest
    # (comma-separated biome names from BIOME_SEEDS).
    WORLD_SEEDS      = (MEADOW_SEED, PLAINS_SEED, TUNDRA_SEED, DESERT_SEED)
    HELDOUT_SEEDS    = (FOREST_SEED,)
    # Lava pools replace some pits/ponds: probability per hazard, by world seed.
    HAZARD_LAVA_PROB = {PLAINS_SEED: 0.6}
    # Sinkholes (5x5, 7 deep, no way out): probability per hazard, by world seed.
    HAZARD_SINKHOLE_PROB = {DESERT_SEED: 0.5}
    LAVA_RADIUS      = 2             # lava pool is (2r+1)^2 cells: 2 -> 5x5 (1 was 3x3)
    REWARD_LAVA      = -30.0         # terminal: walked into lava (outcome "lava")
    BIOME_SEEDS      = BIOME_SEEDS
    HAZARDS          = (1, 3)        # random 3x3 trap holes (dry pit or water-floored pond) per episode
    FILL_WATER       = False         # keep the pond: the world model must learn water/pits = failure
    PIT_DEPTH        = 2.0           # feet this far below the ground 2-3 blocks away = in a pit
    PIT_MAX_STEPS    = 25            # end the episode after this many consecutive pit steps
    PIT_STEP_PENALTY = 0.3           # extra per-step cost while in a pit
    REWARD_PIT       = -30.0         # terminal: stuck in a pit
    PITCH_ALIGN_COEF = 0.3           # potential shaping on pitch error (like AIM_ALIGN_COEF)
    PITCH_LIMIT      = 50.0          # |pitch| beyond this (deg) is penalised per step
    PITCH_PENALTY    = 0.05
    USE_DAMAGE_SIGNAL = True         # Malmo damage event marks a hit (one-hit rabbit kills)

    # ── Observation: pixels + proprioception only ────────────────────────────
    OBS_MODE        = "proprio"      # HuntingEnv drops the target + voxel streams
    EMIT_PRIVILEGED = True           # info["privileged"] = 12-dim target stream
    INPUT_SIZE      = HuntingCFG.PROPRIOCEPTION_SIZE   # 11

    VIDEO_ENABLED  = True
    VIDEO_WIDTH    = 224             # 16x16 DINOv2 patches of 14 px, no resize
    VIDEO_HEIGHT   = 224
    VIDEO_CHANNELS = 3

    # ── Actions: hunting set + jump_forward (natural terrain has 1-block steps)
    ACTIONS = HuntingCFG.HUNTING_ACTIONS + [
        ("jump_forward", ["jump 1", "move 1"], ["jump 0", "move 0"]),   # 11
    ]
    N_ACTIONS = len(ACTIONS)         # 12

    # ── Frozen visual encoder (models/frozen_encoder.py) ─────────────────────
    VWM_INPUT        = "features"    # RSSM consumes frozen-encoder tokens, not pixels
    VIS_ENCODER      = "dinov2_vits14"
    VIS_RES          = 224
    VIS_POOL_GRID    = 4             # patch tokens avg-pooled to GxG (+ CLS) → 17x384
    VIS_TOKEN_PROJ   = 64            # trainable per-token projection before flatten
    VIS_THUMB        = 128           # stored thumbnail size (dream-strip / live viz only);
                                      # decoder upsamples+refines its native 64x64 to this
    VWM_FEAT_DEC_HIDDEN = 512
    VWM_W_THUMB      = 1.0           # thumbnail decoder sees a stop-grad latent

    # Training-time privileged auxiliary target: the latent must predict the
    # rabbit's bearing / distance / in-range (labels from info["privileged"]).
    # Without it the latent ignores the small rabbit and the reward model learns
    # "sprint = reward" from demos where the teacher only sprints when aimed.
    VWM_PRIV_AUX      = True
    VWM_W_PRIV        = 1.0
    VWM_SYMLOG_REWARD = True         # DreamerV3: rare +100 kill no longer spikes the loss

    # ── RSSM (video world model) ─────────────────────────────────────────────
    WM_LATENT_DIM = 32
    USE_GRU       = True
    # KL weighting (2026-11-04, m6warm): the token loss is a per-element MEAN over
    # 17x384 values, so at KL scale 1 the posterior carried only ~1 nat/step and
    # the policy saw the frame through that channel (bearing R2 0.62 -> 0.76 with
    # these values in evaluation/latent_probe.py; live kill rate 12% -> 53%).
    WM_KL_SCALE   = 0.1
    WM_FREE_NATS  = 3.0
    WM_SEQ_LEN    = 32               # longer windows: memory across occlusions
    SEQ_BUFFER_CAPACITY = 40000      # ~13 KB feats + 49 KB thumb (128x128) per step ≈ 2.5 GB

    # ── Lookahead planner (our addition; evaluation only, never used while
    # collecting training data). For each of the actor's top PLAN_CANDIDATES
    # actions, imagine PLAN_SAMPLES futures of PLAN_HORIZON steps (each samples
    # a different stochastic latent = a different pig reaction), score each by
    # discounted predicted reward + critic value at the end, and play the
    # candidate with the best mean (or worst-case) score.
    PLAN_AT_ACT     = False
    PLAN_CANDIDATES = 4
    PLAN_SAMPLES    = 8
    PLAN_HORIZON    = 8
    PLAN_SCORE      = "mean"         # "mean" or "worst" (risk-averse, for hazards)
    # The planner only re-decides aim/attack, and only when the actor would pick
    # one of these; locomotion stays with the (BC-anchored) actor. None = all.
    PLAN_ACTION_SET = None           # only used by PLAN_MODE="argmax"; e.g. ("attack","turn_left",...)
    # Scripted hazard episodes (generate_hunting_demos.py --hazard): a frames demo
    # directory whose episodes end in a pit / lava. HAZARD_WM_FRACTION of each
    # world-model batch, never behaviour-cloned.
    HAZARD_DEMO_PATH   = None
    # How the planner may override the actor: "argmax" = best imagined score if it
    # beats the actor's choice by PLAN_MARGIN; "veto" = only when the actor's
    # choice ends the episode with P >= PLAN_VETO_P (safety filter).
    # 2026-11-04 evals (10 eps, training mix): argmax planner −56 and 0 kills,
    # veto 2 kills/+24, greedy 3 kills/20 eps at +19 -> veto is the default.
    PLAN_MODE   = "veto"
    PLAN_MARGIN = 0.0
    PLAN_VETO_P = 0.5
    HAZARD_WM_FRACTION = 0.1

    # ── Round-3 (2026-11-04) defaults, from the m2/m3/m4 post-mortem ─────────
    # (docs/world_model/results/m2_demo_warmstart/README.md). The fair runs
    # failed because the latent could not see the cow and the reward head keyed
    # on the action one-hot ("attack = reward" even out of range); imagination
    # then rewarded blind attacking. These make the reward head read the state
    # the action leads to, discount reward the heads disagree on, and keep the
    # actor from exploiting what is left.
    VWM_REWARD_ON_NEXT    = True
    VWM_REWARD_ENSEMBLE   = 5
    IMAG_REWARD_PESSIMISM = 1.0
    IMAG_RETURN_NORM      = True
    IMAG_CRITIC_EMA       = 0.02
    IMAG_HORIZON          = 8        # short while the model is young; 15 once the gate passes
    IMAG_ENTROPY_COEF     = 0.001    # with return-normalised advantages (V3 uses 3e-4; 0.03 = uniform)
    BC_PRETRAIN_FREEZE_WM = True     # BC the actor on a frozen world model (features don't drift)
    FAIR_DEMO_BC_COEF     = 0.1      # --fair keeps this anchor for Dreamer (decays to 0)

    # v1: single RSSM, no ensemble/uncertainty machinery (same as hunting_video)
    ENSEMBLE_SIZE     = 1
    WM_PROBABILISTIC  = False
    WM_HORIZON_GATING = False
    INTRINSIC_COEF    = 0.0

    # ── Demos (frames + actions; utils/demo_io.py). Pass --demo-path <dir>. ───
    DEMO_WM_FRACTION    = 0.25
    DEMO_START_FRACTION = 0.3
    DEMO_BC_COEF        = 0.2        # decays to 0 over training (set_progress)
    DEMO_ONLY_SUCCESS   = True       # failed teacher chases are not worth imitating

    # Each checkpoint is ~120 MB (world model + Adam state): keep the newest few.
    KEEP_CHECKPOINTS = 3
    KEEP_CHECKPOINT_EPISODES = (500,)    # cow L0: first 50/50 window — never pruned

    HUNT_LEVEL = 1
    LEVELS = {
        0: {                               # flat walled debug arena, 1 cow
            "MISSION_FILE":     os.path.join(_MISSIONS, "hunting_vision_flat.xml"),
            "SPAWN":            (0.5, 46.0, 0.5),
            "FALL_Y_THRESHOLD": 44.0,
            "TARGET_SPAWN_Y":   46.0,
            "CAGE_HALF":        None,      # the flat mission has its own walls
            "HEIGHTMAP":        None,
            "ARENA_MIN":        -7.0,
            "ARENA_MAX":        7.0,
            "FLEEING_MIN_DIST": 4.0,
            "N_TARGETS":        1,
            "KILLS_TO_END":     1,
            "AUTO_ATTACK":      True,
            "MAX_STEPS":        200,
        },
        1: {},                             # flat meadow + trees, 3 cows, auto-attack on
        2: {"AUTO_ATTACK": False},         # same, the agent aims itself
    }


def apply_hunt_level(cfg):
    """Mutate cfg (a config class) in place for the selected rung.

    MALMO_HUNT_LEVEL (env-server process) overrides cfg.HUNT_LEVEL. Returns the
    level applied. Called by HuntingEnv.__init__ for configs that define LEVELS.
    """
    level = int(os.environ.get("MALMO_HUNT_LEVEL", getattr(cfg, "HUNT_LEVEL", 1)))
    if level not in cfg.LEVELS:
        raise ValueError("Unknown HUNT_LEVEL {0}; choose from {1}".format(
            level, sorted(cfg.LEVELS)))
    # Levels are deltas from the class defaults: snapshot those once so that
    # re-applying a different level in the same process starts clean.
    if "_LEVEL_BASE" not in cfg.__dict__:
        keys = set(k for delta in cfg.LEVELS.values() for k in delta)
        cfg._LEVEL_BASE = dict((k, getattr(cfg, k)) for k in keys)
    for key, value in cfg._LEVEL_BASE.items():
        setattr(cfg, key, value)
    for key, value in cfg.LEVELS[level].items():
        setattr(cfg, key, value)
    cfg.HUNT_LEVEL = level

    prey = os.environ.get("MALMO_HUNT_PREY", getattr(cfg, "PREY", None))
    if prey is not None:
        if prey not in cfg.PREYS:
            raise ValueError("Unknown PREY {0!r}; choose from {1}".format(prey, sorted(cfg.PREYS)))
        for key, value in cfg.PREYS[prey].items():
            setattr(cfg, key, value)
        cfg.PREY = prey
    return level
