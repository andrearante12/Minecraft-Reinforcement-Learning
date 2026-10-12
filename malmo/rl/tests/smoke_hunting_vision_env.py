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


def fake_obs(x=0.5, y=65.0, z=0.5, rabbits=((3.0, 65.0, 4.0, 3.0),)):
    ents = [{"name": "Hunter", "x": x, "y": y, "z": z}]
    for rx, ry, rz, life in rabbits:
        ents.append({"name": HuntingVisionCFG.TARGET_MOB, "x": rx, "y": ry, "z": rz, "life": life})
    return {"XPos": x, "YPos": y, "ZPos": z, "Yaw": 0.0, "Pitch": 0.0,
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
check("hunting XML byte-identical to the pre-change construction",
      env._build_mission_xml([(tx, tz)]) == old)
vec = env._build_obs_vector(fake_obs(y=46.0, rabbits=()))
check("hunting obs is 98-dim", vec.shape == (98,))

print("\n3. Vision obs = proprio only, privileged still computed")
env = HuntingEnv(HuntingVisionCFG)
vec = env._build_obs_vector(fake_obs())
check("hunting_vision obs is 11-dim", vec.shape == (11,) == (HuntingVisionCFG.INPUT_SIZE,))
check("privileged target stream computed (visible, dist>0)",
      env._last_target is not None and env._last_target[11] == 1.0 and env._tgt_dist > 0)

print("\n4. Kill detection with the damage signal")
env = HuntingEnv(HuntingVisionCFG)
env._build_obs_vector(fake_obs(rabbits=((3.0, 65.0, 4.0, 3.0), (10.0, 65.0, 10.0, 3.0))))
prev = (env._tgt_dist, env._tgt_life, env._tgt_visible)
env._damaged = True     # what step() sets when Malmo reports damage dealt
obs = fake_obs(rabbits=((10.0, 65.0, 10.0, 3.0),))   # hit rabbit vanished, another remains
env._build_obs_vector(obs)
_, done, outcome = env._get_reward(obs, *prev)
check("one-hit kill with another rabbit in range -> killed", done and outcome == "killed")
env = HuntingEnv(HuntingVisionCFG)
env._build_obs_vector(fake_obs(rabbits=((3.0, 65.0, 4.0, 3.0), (10.0, 65.0, 10.0, 3.0))))
prev = (env._tgt_dist, env._tgt_life, env._tgt_visible)
obs = fake_obs(rabbits=((10.0, 65.0, 10.0, 3.0),))
env._build_obs_vector(obs)
_, done, outcome = env._get_reward(obs, *prev)
check("body-count drop WITHOUT damage is not a kill", not done)

env = HuntingEnv(HuntingVisionCFG)
env._build_obs_vector(fake_obs(rabbits=((3.0, 65.0, 4.0, 10.0),)))
prev = (env._tgt_dist, env._tgt_life, env._tgt_visible)
obs = fake_obs(rabbits=((3.0, 65.0, 4.0, 0.0),))      # died to the world, no agent damage
env._build_obs_vector(obs)
r, done, outcome = env._get_reward(obs, *prev)
check("prey dying without agent damage is not a kill and earns no hit reward",
      not done and r < 1.0 and not env._damaged)

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
    script.append(dict(fake_obs(y=68.0, rabbits=herd[i:]), _dmg=1.0))   # hit the nearest
    script.append(fake_obs(y=68.0, rabbits=herd[i + 1:]))                # it vanishes
    _, _, done_hit, info = env.step(0)
    _, r, done, info = env.step(0)
    outcomes.append(info["outcome"])
    kill_rewards.append(r)
check("kills 1 and 2 keep the episode going (outcome 'kill'), kill 3 ends it",
      outcomes == ["kill", "kill", "killed"] and done and info["kills"] == 3)
check("every kill pays REWARD_KILL", all(r >= HuntingVisionCFG.REWARD_KILL - 1 for r in kill_rewards))

print("\nALL HUNTING_VISION ENV CHECKS PASSED")
