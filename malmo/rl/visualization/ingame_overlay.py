"""
visualization/ingame_overlay.py
-------------------------------
Turn the world model's imagined future into Minecraft `/particle` commands so it
can be seen IN the game window.

Pure functions (no Malmo import) → fully testable offline. The commands are sent
with EnvClient.chat() to an env_server started with --viz.

Why particles: they are not blocks and not entities, so they never appear in the
voxel-grid or nearby-entity observations the agent/model reads. (They WOULD
appear in video frames — keep the overlay off while collecting training frames.)

Legend (default style):
    flame          imagined PIG path (every `stride`-th step, body height)
    endRod         "ghost pig": 4 top corners + centre where the pig is imagined at `ghost_step`
    happyVillager  imagined AGENT path (at the agent's feet)
    witchMagic     each ensemble member's guess of the pig at `ghost_step` (the fan)

1.11.2 syntax: /particle <name> x y z xd yd zd speed [count] [mode]
Names are configurable (STYLE) because they vary by Minecraft version; run
`live_imagination.py --probe` to see them all around the spawn.
"""

import time
import numpy as np

from evaluation.imagination_vs_reality import decode_positions

STYLE = {
    "pig_path":   "flame",
    "ghost":      "endRod",
    "agent_path": "happyVillager",
    "members":    "witchMagic",
}


def particle(name, x, y, z, count=1):
    return "/particle {n} {x:.2f} {y:.2f} {z:.2f} 0 0 0 0 {c} force".format(
        n=name, x=x, y=y, z=z, c=count)


def build_commands(imag_obs_seq, cfg, members_obs=None, style=None, stride=2,
                   ghost_step=3, max_cmds=16, pig_body_h=0.6, agent_h=0.2):
    """Imagined obs sequence (index 0 = now) → list of /particle command strings.

    imag_obs_seq: (T,D) ensemble-mean imagined observations starting at the current real obs
    members_obs:  optional (K,T,D) per-member sequences (draws the fan at ghost_step)
    Priority when over budget: ghost > pig path > fan > agent path.
    """
    st = dict(STYLE, **(style or {}))
    ag, pg = decode_positions(imag_obs_seq, cfg)
    T = len(imag_obs_seq)
    gs = min(ghost_step, T - 1)
    ghost, pigp, fan, agp = [], [], [], []

    if pg is not None:
        cx, cy, cz = pg[gs]
        for dx, dz in ((-.45, -.45), (.45, -.45), (-.45, .45), (.45, .45)):
            ghost.append(particle(st["ghost"], cx + dx, cy + 0.9, cz + dz))
        ghost.append(particle(st["ghost"], cx, cy + 0.45, cz))
        for t in range(stride, T, stride):
            pigp.append(particle(st["pig_path"], pg[t][0], pg[t][1] + pig_body_h, pg[t][2]))
        for m in (members_obs if members_obs is not None else []):
            mp = decode_positions(np.asarray(m)[gs:gs + 1], cfg)[1][0]
            fan.append(particle(st["members"], mp[0], mp[1] + pig_body_h, mp[2]))
    for t in range(stride, T, stride):
        agp.append(particle(st["agent_path"], ag[t][0], ag[t][1] + agent_h, ag[t][2]))

    return (ghost + pigp + fan + agp)[:max_cmds]


def probe_commands(origin, style=None):
    """One marker per style name in a row in front of `origin` (x,y,z), for eyeballing
    which particle names exist in the running Minecraft version."""
    st = dict(STYLE, **(style or {}))
    x, y, z = origin
    return [particle(name, x + 1.0 * i - 1.5, y + 1.0, z + 3.0, count=8)
            for i, name in enumerate(st.values())]


class Throttle:
    """Cap commands per second (Minecraft kicks non-op players for chat spam)."""

    def __init__(self, max_per_sec=40.0, clock=time.time):
        self.rate, self.clock = max_per_sec, clock
        self.tokens, self.last = max_per_sec, clock()

    def take(self, n):
        """Return how many of `n` commands may be sent now."""
        now = self.clock()
        self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
        self.last = now
        k = int(min(n, self.tokens))
        self.tokens -= k
        return k
