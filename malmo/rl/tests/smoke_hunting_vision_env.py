"""
tests/smoke_hunting_vision_env.py
----------------------------------
Offline (no Minecraft) checks of the env-server side of hunting_vision, run in
the malmo conda env (Python 3.6 — no f-strings here):

    conda run -n malmo python malmo/rl/tests/smoke_hunting_vision_env.py

Checks:
  - every rung's mission XML (cage + DrawEntity + seed) passes Malmo's schema
    validation (MissionSpec(xml, True)), with the right entity count
  - the plain `hunting` env's per-episode XML is byte-identical to the
    pre-change construction (default path untouched)
  - levels are deltas from class defaults: L3 then L1 leaves no L3 residue
  - obs vector: 98 dims for hunting, 11 (proprio only) for hunting_vision,
    while the privileged target stream is still computed
  - kill detection: the Malmo damage event + a body-count drop ends the
    episode as "killed" (one-hit rabbit kills that never show life 0)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # malmo/rl
sys.path.insert(0, ROOT)

import numpy as np

from envs.hunting_env import HuntingEnv, MalmoPython
from training.configs.hunting_cfg import HuntingCFG
from training.configs.hunting_vision_cfg import HuntingVisionCFG, apply_hunt_level


def check(name, cond):
    print("[{0}] {1}".format("OK" if cond else "FAIL", name))
    if not cond:
        raise SystemExit("Smoke test failed: " + name)


def fake_obs(x=0.5, y=65.0, z=0.5, rabbits=((3.0, 65.0, 4.0, 3.0),), killed=0):
    """`killed` = the player's MobsKilled stat. Entity ids are stable per spawn
    position (positions don't move in these tests)."""
    ents = [{"name": "Hunter", "x": x, "y": y, "z": z}]
    for rx, ry, rz, life in rabbits:
        ents.append({"name": HuntingVisionCFG.TARGET_MOB, "x": rx, "y": ry, "z": rz, "life": life,
                     "id": "prey_{0:g}_{1:g}".format(rx, rz)})
    return {"XPos": x, "YPos": y, "ZPos": z, "Yaw": 0.0, "Pitch": 0.0, "MobsKilled": killed,
            "OnGround": True, "Food": 20, "Life": 20, "entities": ents,
            "LineOfSight": {}, "floor3x3": ["air"] * 75}


print("1. Mission XML per rung")
os.environ.pop("MALMO_HUNT_LEVEL", None)
for level in sorted(HuntingVisionCFG.LEVELS):
    os.environ["MALMO_HUNT_LEVEL"] = str(level)
    env = HuntingEnv(HuntingVisionCFG)
    spawns = [env._choose_target_spawn() for _ in range(HuntingVisionCFG.N_TARGETS)]
    xml = env._build_mission_xml(spawns)
    try:
        MalmoPython.MissionSpec(xml, True)
        valid = True
    except Exception as e:
        print("   schema error:", e)
        valid = False
    mob = HuntingVisionCFG.TARGET_MOB
    check("L{0} XML validates ({1} x {2}, cage={3})".format(
        level, HuntingVisionCFG.N_TARGETS, mob, HuntingVisionCFG.CAGE_HALF), valid)
    check("L{0} DrawEntity count".format(level),
          xml.count('type="{0}"/>'.format(mob)) == HuntingVisionCFG.N_TARGETS)
    check("L{0} damage reward is for the prey".format(level), '<Mob type="{0}"'.format(mob) in xml)
    h = HuntingVisionCFG.CAGE_HALF
    style = HuntingVisionCFG.CAGE_STYLE if HuntingVisionCFG.HEIGHTMAP else "bedrock"
    fence = style == "fence"
    marker = 'type="{0}"'.format("bedrock" if style == "bedrock" else style)
    check("L{0} cage drawn iff CAGE_HALF ({1})".format(level, style),
          (marker in xml) == bool(h))
    if h and fence:
        check("L{0} fence ring has {1} posts, no bedrock".format(level, 8 * h),
              xml.count('type="fence"') == 8 * h and 'type="bedrock"' not in xml)
    if HuntingVisionCFG.WORLD_SEED is not None and "DefaultWorldGenerator" in xml:
        check("L{0} seed substituted".format(level),
              'seed="{0}"'.format(HuntingVisionCFG.WORLD_SEED) in xml)
    if HuntingVisionCFG.HEIGHTMAP:
        sx, sy, sz = HuntingVisionCFG.SPAWN
        check("L{0} spawn from heightmap is on standable ground ({1})".format(level, HuntingVisionCFG.SPAWN),
              (int(sx - 0.5), int(sz - 0.5)) in env._hm_ok and 60 < sy < 100)
        check("L{0} Placement uses the heightmap spawn".format(level),
              '<Placement x="{0}" y="{1}" z="{2}"'.format(sx, sy, sz) in xml)
        ok = all((int(x - 0.5), int(z - 0.5)) in env._hm_ok for x, z in
                 [env._choose_target_spawn() for _ in range(200)])
        check("L{0} prey spawns only on standable columns".format(level), ok)
os.environ["MALMO_HUNT_LEVEL"] = "0"
apply_hunt_level(HuntingVisionCFG)
os.environ["MALMO_HUNT_LEVEL"] = "1"
apply_hunt_level(HuntingVisionCFG)
check("default prey is cow (Cow, 10 HP)",
      HuntingVisionCFG.PREY == "cow" and HuntingVisionCFG.TARGET_MOB == "Cow"
      and HuntingVisionCFG.TARGET_MAX_LIFE == 10.0)
os.environ["MALMO_HUNT_PREY"] = "rabbit"
apply_hunt_level(HuntingVisionCFG)
check("MALMO_HUNT_PREY=rabbit switches prey", HuntingVisionCFG.TARGET_MOB == "Rabbit")
del os.environ["MALMO_HUNT_PREY"]
apply_hunt_level(HuntingVisionCFG)
check("L0 -> L1 leaves no residue (meadow seed, 3 cows, CAGE_HALF=21, AUTO_ATTACK)",
      HuntingVisionCFG.N_TARGETS == 3 and HuntingVisionCFG.KILLS_TO_END == 3
      and HuntingVisionCFG.CAGE_HALF == 21 and HuntingVisionCFG.WORLD_SEED == 523467673
      and "523467673" in HuntingVisionCFG.HEIGHTMAP
      and HuntingVisionCFG.AUTO_ATTACK is True and HuntingVisionCFG.MAX_STEPS == 900)

print("\n2. Default hunting path untouched")
env = HuntingEnv(HuntingCFG)
tx, tz = 1.25, -3.5
old = env._mission_xml_base.replace(
    '</DrawingDecorator>',
    '    <DrawEntity x="{x:.1f}" y="46.0" z="{z:.1f}" type="{mob}"/>\n'.format(
        x=tx, z=tz, mob=HuntingCFG.TARGET_MOB) + '      </DrawingDecorator>')
# The only intended change: the cfg's MISSION_TIME_LIMIT_MS replaces the XML's 60 s.
import re
old = re.sub(r'(<ServerQuitFromTimeUp[^>]*timeLimitMs=")[^"]*"',
             r'\g<1>{0}"'.format(int(HuntingCFG.MISSION_TIME_LIMIT_MS)), old)
check("hunting XML byte-identical to the pre-change construction (+ time limit)",
      env._build_mission_xml([(tx, tz)]) == old)
vec = env._build_obs_vector(fake_obs(y=46.0, rabbits=()))
check("hunting obs is 98-dim", vec.shape == (98,))

print("\n3. Vision obs = proprio only, privileged still computed")
env = HuntingEnv(HuntingVisionCFG)
vec = env._build_obs_vector(fake_obs())
check("hunting_vision obs is 11-dim", vec.shape == (11,) == (HuntingVisionCFG.INPUT_SIZE,))
check("privileged target stream computed (visible, dist>0)",
      env._last_target is not None and env._last_target[11] == 1.0 and env._tgt_dist > 0)

print("\n3b. Natural pigs count as prey (EXTRA_PREY_MOBS)")
env = HuntingEnv(HuntingVisionCFG)
o = fake_obs(rabbits=())
o["entities"].append({"name": "Pig", "x": 2.0, "y": 65.0, "z": 3.0, "life": 10.0})
o["entities"].append({"name": "Sheep", "x": 1.0, "y": 65.0, "z": 1.0, "life": 8.0})
env._build_obs_vector(o)
check("a lone pig is targeted (sheep ignored)", env._tgt_visible and env._n_targets == 1
      and abs(env._tgt_dist - np.hypot(1.5, 2.5)) < 1e-4)
check("damage reward XML lists Cow and Pig",
      env._build_mission_xml([(1.5, 2.5)]).count('<Mob type=') == 2)

print("\n3c. Pit = cannot walk out (hillsides are not pits)")
import random
os.environ["MALMO_WORLD_SEEDS"], os.environ["MALMO_SINK_PROB"] = "forest", "1"
saved_hz = HuntingVisionCFG.HAZARDS
HuntingVisionCFG.HAZARDS = (1, 1)
env = HuntingEnv(HuntingVisionCFG)
random.seed(1)
env._pick_world()


def _in_pit(x, z):
    y = env._feet_y(x + 0.5, z + 0.5)
    return env._pit_depth(x + 0.5, z + 0.5, y) >= HuntingVisionCFG.PIT_DEPTH and env._trapped(x + 0.5, z + 0.5, y)


sx, sy, sz = HuntingVisionCFG.SPAWN
check("hilly forest spawn is not a pit", not _in_pit(int(sx // 1), int(sz // 1)))
open_flagged = sum(_in_pit(x, z) for x, z in env._hm_ok)
check("under 1% of forest open ground counts as a pit ({0}/{1})".format(open_flagged, len(env._hm_ok)),
      open_flagged < 0.01 * len(env._hm_ok))
check("every sinkhole floor cell is a pit", all(_in_pit(x, z) for x, z in env._hazard_cells))
HuntingVisionCFG.HAZARDS = saved_hz
for k in ("MALMO_WORLD_SEEDS", "MALMO_SINK_PROB"):
    os.environ.pop(k, None)

print("\n4. Kill oracle (MobsKilled + entity id)")
TWO = ((3.0, 65.0, 4.0, 3.0), (10.0, 65.0, 10.0, 3.0))   # near (in reach) + far


def kill_case(before, after, killed_after, damaged=True):
    env = HuntingEnv(HuntingVisionCFG)
    env._build_obs_vector(fake_obs(rabbits=before, killed=0))
    prev = (env._tgt_dist, env._tgt_life, env._tgt_visible)
    env._damaged = damaged
    obs = fake_obs(rabbits=after, killed=killed_after)
    env._build_obs_vector(obs)
    return env._get_reward(obs, *prev)


_, done, outcome = kill_case(TWO, TWO[1:], killed_after=1)
check("MobsKilled +1 and the in-reach prey is gone -> killed", done and outcome == "killed")
_, done, outcome = kill_case(TWO, ((3.0, 65.0, 4.0, 0.0),) + TWO[1:], killed_after=1)
check("MobsKilled +1 and the in-reach prey still listed at 0 health (death anim) -> killed",
      done and outcome == "killed")
_, done, _ = kill_case(TWO, TWO[1:], killed_after=0)
check("hit prey vanishes (wandered off / despawned) without MobsKilled -> NOT a kill", not done)
_, done, _ = kill_case(TWO, TWO[:1], killed_after=0)
check("far prey leaves sensor range -> NOT a kill", not done)
_, done, _ = kill_case(TWO, TWO, killed_after=1)
check("MobsKilled +1 but every prey still there (killed a sheep) -> NOT a kill", not done)
_, done, _ = kill_case(TWO, TWO[:1], killed_after=1)
check("MobsKilled +1 but only an out-of-reach prey vanished -> NOT a kill", not done)
r, done, _ = kill_case(((3.0, 65.0, 4.0, 10.0),), ((3.0, 65.0, 4.0, 0.0),), killed_after=0, damaged=False)
check("prey dying to the world (no MobsKilled) is not a kill", not done)

print("\n5. Multi-kill episodes (KILLS_TO_END=3), step() with a mocked Malmo")


class _Host(object):
    def sendCommand(self, c):
        pass


class _WS(object):
    is_mission_running = True


env = HuntingEnv(HuntingVisionCFG)
env._agent_host = _Host()
env._take_action = lambda a: None
herd = [(3.0, 68.0, 4.0, 10.0), (8.0, 68.0, 8.0, 10.0), (-9.0, 68.0, 5.0, 10.0)]
script = []


def _obs_dict(timeout=3.0):
    o = script.pop(0)
    env._dmg_signal += o.pop("_dmg", 0.0)
    return o, _WS()


env._get_obs_dict = _obs_dict
env._build_obs_vector(fake_obs(y=68.0, rabbits=herd))
outcomes, kill_rewards = [], []
for i in range(3):
    ax, az = herd[i][0] - 1.0, herd[i][2]                                # walked up to animal i
    script.append(dict(fake_obs(x=ax, y=68.0, z=az, rabbits=herd[i:], killed=i), _dmg=1.0))  # hit it
    script.append(fake_obs(x=ax, y=68.0, z=az, rabbits=herd[i + 1:], killed=i + 1))         # it dies
    _, _, done_hit, info = env.step(0)
    _, r, done, info = env.step(0)
    outcomes.append(info["outcome"])
    kill_rewards.append(r)
check("kills 1 and 2 keep the episode going (outcome 'kill'), kill 3 ends it",
      outcomes == ["kill", "kill", "killed"] and done and info["kills"] == 3)
check("every kill pays REWARD_KILL", all(r >= HuntingVisionCFG.REWARD_KILL - 1 for r in kill_rewards))

print("\nALL HUNTING_VISION ENV CHECKS PASSED")
