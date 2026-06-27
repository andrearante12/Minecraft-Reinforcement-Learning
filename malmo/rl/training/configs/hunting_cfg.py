"""
training/configs/hunting_cfg.py
-------------------------------
Environment config for the hunting task: the agent must locate, aim at,
approach, and attack a passive animal (default Pig) until it dies.

This is the headline showcase for the world-model agent — a long-horizon,
sparse-reward task. The TARGET_MODE knob (penned / wandering / fleeing)
controls how predictable the moving target is, which is the central research
variable for the deterministic-vs-stochastic world-model study.

Inherits shared + world-model hyperparameters from BaseCFG.
"""

import os
from training.configs.base_cfg import BaseCFG


class HuntingCFG(BaseCFG):
    # ── Mission ──────────────────────────────────────────────────────────────
    MISSION_FILE = os.path.join(
        BaseCFG.ROOT_DIR, "envs", "hunting", "missions", "hunting.xml"
    )
    MALMO_PORT = 10000

    # ── Agent spawn ──────────────────────────────────────────────────────────
    SPAWN    = (0.5, 46.0, 0.5)
    GOAL_POS = (0.5, 46.0, 4.5)   # nominal target location (logging only)

    # ── Target animal ────────────────────────────────────────────────────────
    TARGET_MOB      = "Pig"
    TARGET_MAX_LIFE = 10.0        # Pig max health (for normalisation)
    ATTACK_RANGE    = 3.0         # blocks; within this an attack can land

    # Predictability knob (the research variable):
    #   penned    — fixed, close spawn (near-deterministic pursuit)
    #   wandering — random spawn anywhere in the arena
    #   fleeing   — random far spawn (animal more likely to wander out of reach)
    TARGET_MODE   = "penned"
    PENNED_DZ     = 4.0           # penned: target this many blocks ahead (+Z)
    ARENA_MIN     = -7.0          # interior bounds for random target spawn
    ARENA_MAX     = 7.0
    FLEEING_MIN_DIST = 6.0        # fleeing: target spawns at least this far away

    # ── Episode termination ──────────────────────────────────────────────────
    FALL_Y_THRESHOLD = 44.0       # defensive (arena is walled/flat)
    MAX_STEPS        = 200
    STEP_DURATION    = 0.15

    # ── Voxel grid: x[-2:+2]=5, y[-1:+1]=3, z[-2:+2]=5 → 75 (flat arena: walls) ─
    GRID_X    = 5
    GRID_Y    = 3
    GRID_Z    = 5
    GRID_SIZE = GRID_X * GRID_Y * GRID_Z  # 75

    BLOCK_ENCODING = {"air": 0, "stone": 1, "grass": 1, "dirt": 1}

    # ── Observation ──────────────────────────────────────────────────────────
    # Proprioception (11): onGround, yaw, pitch, vel(3), pos-rel-spawn(3),
    #                      food, health
    # Target (12):         rel pos(3), rel vel(3), distance, heading_error,
    #                      los_hit, in_range, target_life, target_visible
    PROPRIOCEPTION_SIZE = 11
    GOAL_DELTA_SIZE     = 12       # repurposed as the moving-target stream
    INPUT_SIZE          = PROPRIOCEPTION_SIZE + GOAL_DELTA_SIZE + GRID_SIZE  # 98

    DIST_SCALE = 20.0             # normaliser for target distance

    # ── Actions (11 discrete) ────────────────────────────────────────────────
    HUNTING_ACTIONS = [
        ("move_forward",   ["move 1"],              ["move 0"]),               # 0
        ("move_backward",  ["move -1"],             ["move 0"]),               # 1
        ("strafe_left",    ["strafe -1"],           ["strafe 0"]),             # 2
        ("strafe_right",   ["strafe 1"],            ["strafe 0"]),             # 3
        ("sprint_forward", ["sprint 1", "move 1"],  ["sprint 0", "move 0"]),   # 4
        ("turn_left",      ["turn -0.5"],           ["turn 0"]),               # 5
        ("turn_right",     ["turn 0.5"],            ["turn 0"]),               # 6
        ("look_up",        ["pitch -0.5"],          ["pitch 0"]),              # 7
        ("look_down",      ["pitch 0.5"],           ["pitch 0"]),              # 8
        ("attack",         ["attack 1"],            ["attack 0"]),             # 9
        ("no_op",          [],                      []),                       # 10
    ]
    ACTIONS   = HUNTING_ACTIONS
    N_ACTIONS = len(ACTIONS)  # 11

    # ── Rewards ──────────────────────────────────────────────────────────────
    REWARD_HIT          = +5.0    # per point of target health removed
    REWARD_KILL         = +100.0  # dominant terminal success signal
    REWARD_APPROACH_COEF = +1.0   # per block of distance closed to the target
    REWARD_AIM          = +0.05   # per step the crosshair is on the animal
    REWARD_STEP_PENALTY = -0.02   # mild speed pressure
    REWARD_TIMEOUT      = -5.0
    REWARD_FELL         = -10.0
    APPROACH_CLIP       = 2.0     # clamp per-step approach delta (blocks)

    # ── Hyperparameter notes ─────────────────────────────────────────────────
    # World-model defaults come from WorldModelCFG. Long-horizon sparse task:
    TOTAL_EPISODES = 8000
    SAVE_EVERY     = 25
