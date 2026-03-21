"""
training/curriculum.py
----------------------
Curriculum scheduler for training across multiple environments.

Supports three modes:
  - sequential: fixed stages with episode counts
  - weighted: batched round-robin with weights per env (e.g. window=10,
    weight=0.4 → 4 consecutive episodes on that env per window)
  - adaptive: performance-gated progression — advance to the next stage
    when the agent hits a target success rate on the current one

All environments in a curriculum must share the same INPUT_SIZE and N_ACTIONS.
"""

import json
from collections import deque


class CurriculumScheduler:
    def __init__(self, schedule_dict, env_registry):
        self.mode = schedule_dict["mode"]
        if self.mode not in ("sequential", "weighted", "adaptive"):
            raise ValueError("Unknown curriculum mode: {0}".format(self.mode))

        # Collect all env names referenced in the schedule
        if self.mode == "sequential":
            self.stages = schedule_dict["stages"]
            env_names = [s["env"] for s in self.stages]
        elif self.mode == "adaptive":
            self.stages = schedule_dict["stages"]
            env_names = [s["env"] for s in self.stages]
            # Set defaults for each stage
            for s in self.stages:
                s.setdefault("window", 50)
                s.setdefault("min_episodes", 100)
                s.setdefault("max_episodes", 2000)
                # Last stage doesn't need target_success_rate
                s.setdefault("target_success_rate", None)
            # Internal adaptive state
            self._current_stage = 0
            self._stage_episode_count = 0
            self._stage_outcomes = deque(maxlen=self.stages[0]["window"])
        else:
            self.envs = schedule_dict["envs"]
            self._total_episodes = schedule_dict["total_episodes"]
            self._window = schedule_dict.get("window", 10)
            env_names = [e["env"] for e in self.envs]

            # Pre-compute the episode schedule within one window.
            # Weights are converted to consecutive episode counts.
            # e.g. weights [0.4, 0.6] with window=10 → [4, 6] episodes
            total_weight = sum(e["weight"] for e in self.envs)
            raw_counts = [(e["weight"] / total_weight) * self._window for e in self.envs]

            # Round with largest-remainder method to ensure counts sum to window
            floor_counts = [int(c) for c in raw_counts]
            remainders = [(raw_counts[i] - floor_counts[i], i) for i in range(len(raw_counts))]
            shortfall = self._window - sum(floor_counts)
            remainders.sort(reverse=True)
            for j in range(shortfall):
                floor_counts[remainders[j][1]] += 1

            # Build window schedule: consecutive blocks per env
            self._window_schedule = []
            for i, e in enumerate(self.envs):
                self._window_schedule.extend([e["env"]] * floor_counts[i])

        # Validate all env names exist in registry
        for name in env_names:
            if name not in env_registry:
                raise ValueError("Curriculum references unknown env: {0}".format(name))

        # Validate all envs share INPUT_SIZE and N_ACTIONS
        cfgs = [env_registry[name][1] for name in env_names]
        input_sizes = set(c.INPUT_SIZE for c in cfgs)
        n_actions = set(c.N_ACTIONS for c in cfgs)
        if len(input_sizes) > 1:
            raise ValueError("Curriculum envs have mismatched INPUT_SIZE: {0}".format(input_sizes))
        if len(n_actions) > 1:
            raise ValueError("Curriculum envs have mismatched N_ACTIONS: {0}".format(n_actions))

    def env_for_episode(self, episode):
        """Return the env name for a given episode number (1-based)."""
        if self.mode == "sequential":
            cumulative = 0
            for stage in self.stages:
                cumulative += stage["episodes"]
                if episode <= cumulative:
                    return stage["env"]
            # Past the end — stay on last stage
            return self.stages[-1]["env"]
        elif self.mode == "adaptive":
            return self.stages[self._current_stage]["env"]
        else:
            # Weighted: batched round-robin within repeating windows
            pos = (episode - 1) % self._window
            return self._window_schedule[pos]

    def total_episodes(self):
        if self.mode == "sequential":
            return sum(s["episodes"] for s in self.stages)
        elif self.mode == "adaptive":
            return sum(s["max_episodes"] for s in self.stages)
        else:
            return self._total_episodes

    def all_env_names(self):
        """Return unique env names in the curriculum."""
        if self.mode in ("sequential", "adaptive"):
            seen = []
            for s in self.stages:
                if s["env"] not in seen:
                    seen.append(s["env"])
            return seen
        else:
            seen = []
            for e in self.envs:
                if e["env"] not in seen:
                    seen.append(e["env"])
            return seen

    # ── Adaptive mode methods ────────────────────────────────────────────────

    def report_outcome(self, outcome):
        """Report an episode outcome. Only acts in adaptive mode."""
        if self.mode != "adaptive":
            return
        self._stage_outcomes.append(outcome == "landed")
        self._stage_episode_count += 1
        self._check_promotion()

    def _check_promotion(self):
        """Check if the agent should advance to the next stage."""
        # Already on last stage — no promotion possible
        if self._current_stage >= len(self.stages) - 1:
            return
        stage = self.stages[self._current_stage]
        if self._stage_episode_count < stage["min_episodes"]:
            return
        success_rate = self.current_success_rate()
        target = stage.get("target_success_rate")
        if (target is not None and success_rate >= target) or \
                self._stage_episode_count >= stage["max_episodes"]:
            self._advance_stage(success_rate)

    def _advance_stage(self, success_rate):
        """Move to the next curriculum stage."""
        old_env = self.stages[self._current_stage]["env"]
        self._current_stage += 1
        new_stage = self.stages[self._current_stage]
        self._stage_episode_count = 0
        self._stage_outcomes = deque(maxlen=new_stage["window"])
        print()
        print("=" * 60)
        print("CURRICULUM PROMOTION")
        print("  {0} -> {1}".format(old_env, new_stage["env"]))
        print("  Success rate achieved: {0:.1%}".format(success_rate))
        print("  Now on stage {0}/{1}".format(
            self._current_stage + 1, len(self.stages)))
        print("=" * 60)
        print()

    def current_success_rate(self):
        """Return rolling success rate for the current adaptive stage."""
        if not self._stage_outcomes:
            return 0.0
        return sum(self._stage_outcomes) / len(self._stage_outcomes)

    def current_stage_name(self):
        """Return the current adaptive stage env name."""
        if self.mode != "adaptive":
            return None
        return self.stages[self._current_stage]["env"]

    def is_complete(self):
        """Return True if the adaptive curriculum is finished."""
        if self.mode != "adaptive":
            return False
        if self._current_stage >= len(self.stages):
            return True
        if self._current_stage == len(self.stages) - 1:
            return self._stage_episode_count >= self.stages[self._current_stage]["max_episodes"]
        return False

    def current_stage_info(self):
        """Return a dict with current stage progress (for logging)."""
        if self.mode != "adaptive":
            return {}
        stage = self.stages[self._current_stage]
        return {
            "stage": self._current_stage + 1,
            "total_stages": len(self.stages),
            "env": stage["env"],
            "episode_on_stage": self._stage_episode_count,
            "max_episodes": stage["max_episodes"],
            "success_rate": self.current_success_rate(),
            "target": stage.get("target_success_rate"),
        }

    # ── State dict for checkpoint save/resume ────────────────────────────────

    def state_dict(self):
        """Return serializable state for checkpoint. None for non-adaptive."""
        if self.mode != "adaptive":
            return None
        return {
            "mode": self.mode,
            "current_stage": self._current_stage,
            "stage_episode_count": self._stage_episode_count,
            "stage_outcomes": list(self._stage_outcomes),
        }

    def load_state_dict(self, d):
        """Restore adaptive state from a checkpoint dict."""
        if self.mode != "adaptive" or d is None:
            return
        self._current_stage = d["current_stage"]
        self._stage_episode_count = d["stage_episode_count"]
        window = self.stages[self._current_stage]["window"]
        self._stage_outcomes = deque(d["stage_outcomes"], maxlen=window)

    @staticmethod
    def from_json(path, env_registry):
        """Load a curriculum schedule from a JSON file."""
        with open(path, "r") as f:
            schedule_dict = json.load(f)
        return CurriculumScheduler(schedule_dict, env_registry)

    @staticmethod
    def single_env(env_name, total_episodes, env_registry):
        """Create a trivial single-stage curriculum for backward compat."""
        schedule_dict = {
            "mode": "sequential",
            "stages": [{"env": env_name, "episodes": total_episodes}],
        }
        return CurriculumScheduler(schedule_dict, env_registry)
