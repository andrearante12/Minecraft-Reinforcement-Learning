"""
training/curriculum.py
----------------------
Curriculum scheduler for training across multiple environments.

Supports two modes:
  - sequential: fixed stages with episode counts
  - weighted: batched round-robin with weights per env (e.g. window=10,
    weight=0.4 → 4 consecutive episodes on that env per window)

All environments in a curriculum must share the same INPUT_SIZE and N_ACTIONS.
"""

import json


class CurriculumScheduler:
    def __init__(self, schedule_dict, env_registry):
        self.mode = schedule_dict["mode"]
        if self.mode not in ("sequential", "weighted"):
            raise ValueError("Unknown curriculum mode: {0}".format(self.mode))

        # Collect all env names referenced in the schedule
        if self.mode == "sequential":
            self.stages = schedule_dict["stages"]
            env_names = [s["env"] for s in self.stages]
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
        else:
            # Weighted: batched round-robin within repeating windows
            pos = (episode - 1) % self._window
            return self._window_schedule[pos]

    def total_episodes(self):
        if self.mode == "sequential":
            return sum(s["episodes"] for s in self.stages)
        else:
            return self._total_episodes

    def all_env_names(self):
        """Return unique env names in the curriculum."""
        if self.mode == "sequential":
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
