"""
utils/logger.py
---------------
Logs training metrics to CSV and console.
"""

import os
import csv
import time
from collections import deque


class Logger:
    def __init__(self, log_dir, run_name):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base      = os.path.join(log_dir, "{0}_{1}".format(run_name, timestamp))

        # Episode log — fixed columns
        self.ep_path   = base + "_episodes.csv"
        self.ep_file   = open(self.ep_path, "w", newline="")
        self.ep_writer = csv.writer(self.ep_file)
        self.ep_writer.writerow(["episode", "reward", "steps", "outcome", "env", "timestamp"])

        # Update log — columns written dynamically on first call
        self.upd_path    = base + "_updates.csv"
        self.upd_file    = open(self.upd_path, "w", newline="")
        self.upd_writer  = csv.writer(self.upd_file)
        self._upd_header = False   # header written on first log_update call

        # Trajectory log — header written on first init_trajectory() call
        self.traj_path   = base + "_trajectories.csv"
        self.traj_file   = open(self.traj_path, "w", newline="")
        self.traj_writer = csv.writer(self.traj_file)
        self._traj_init  = False

        # Rolling window for console summary
        self.window        = 100
        self.ep_rewards    = deque(maxlen=self.window)
        self.ep_steps      = deque(maxlen=self.window)
        self.ep_outcomes   = deque(maxlen=self.window)
        self.ep_envs       = deque(maxlen=self.window)
        self.episode_count = 0
        self.update_count  = 0
        self.start_time    = time.time()

        print("Logging to:")
        print("  Episodes:     {0}".format(self.ep_path))
        print("  Updates:      {0}".format(self.upd_path))
        print("  Trajectories: {0}".format(self.traj_path))

    def init_trajectory(self, env_name, cfg):
        """
        Write the comment-header block and column header that data_loader.py expects.
        Call once after the logger is created and the initial env is known.

        Block geometry is omitted here — data_loader.py supplements from the
        mission XML via the env name in the # env: line.
        """
        self.traj_file.write("# env: {0}\n".format(env_name))
        self.traj_file.write("# spawn: {0},{1},{2}\n".format(*cfg.SPAWN))
        self.traj_file.write("# goal: {0},{1},{2}\n".format(*cfg.GOAL_POS))
        self.traj_writer.writerow([
            "episode", "step", "x", "y", "z",
            "yaw", "pitch", "on_ground",
            "action", "reward", "done", "outcome", "env",
        ])
        self.traj_file.flush()
        self._traj_init = True

    def log_step(self, episode, step, info, reward, done, env_name=""):
        """
        Append one row to the trajectory CSV.

        Uses .get() for yaw/pitch/on_ground so the env_client error-fallback
        path (which returns a minimal info dict) is handled gracefully.
        """
        if not self._traj_init:
            return
        pos = info.get("pos", (0.0, 0.0, 0.0))
        self.traj_writer.writerow([
            episode,
            step,
            round(float(pos[0]), 4),
            round(float(pos[1]), 4),
            round(float(pos[2]), 4),
            round(float(info.get("yaw",      0.0)), 2),
            round(float(info.get("pitch",    0.0)), 2),
            int(info.get("on_ground", 1)),
            info.get("action", "none"),
            round(float(reward), 4),
            int(done),
            info.get("outcome", "alive"),
            env_name,
        ])
        self.traj_file.flush()

    def log_episode(self, episode, reward, steps, outcome, env_name=""):
        self.ep_writer.writerow([episode, round(reward, 4), steps, outcome,
                                  env_name, time.strftime("%H:%M:%S")])
        self.ep_file.flush()
        self.ep_rewards.append(reward)
        self.ep_steps.append(steps)
        self.ep_outcomes.append(outcome)
        self.ep_envs.append(env_name)
        self.episode_count = episode

    def log_update(self, **losses):
        """
        Accepts whatever keys the algorithm's update() returns.
        Header is written on the first call using the keys from that call.
        """
        self.update_count += 1
        if not self._upd_header:
            self.upd_writer.writerow(["update"] + list(losses.keys()))
            self._upd_header = True
        self.upd_writer.writerow(
            [self.update_count] + [round(v, 6) for v in losses.values()]
        )
        self.upd_file.flush()

    def print_summary(self, every=10):
        if self.episode_count % every != 0 or self.episode_count == 0:
            return

        n          = len(self.ep_rewards)
        mean_rew   = sum(self.ep_rewards) / n
        mean_steps = sum(self.ep_steps) / n
        n_landed   = sum(1 for o in self.ep_outcomes if o == "landed")
        n_fell     = sum(1 for o in self.ep_outcomes if o == "fell")
        n_timeout  = sum(1 for o in self.ep_outcomes if o == "timeout")
        elapsed    = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("Episode {0:>5} | Elapsed: {1:.0f}s".format(self.episode_count, elapsed))
        print("  Reward (last {0}): mean={1:.2f}  min={2:.2f}  max={3:.2f}".format(
            n, mean_rew, min(self.ep_rewards), max(self.ep_rewards)))
        print("  Steps  (last {0}): mean={1:.1f}".format(n, mean_steps))
        print("  Outcomes: landed={0}  fell={1}  timeout={2}".format(
            n_landed, n_fell, n_timeout))
        print("  Success rate: {0:.1f}%".format(100 * n_landed / n))

        # Per-env breakdown (only if multiple envs seen)
        unique_envs = set(self.ep_envs)
        unique_envs.discard("")
        if len(unique_envs) > 1:
            print("  Per-env:")
            for env_name in sorted(unique_envs):
                env_rewards = [r for r, e in zip(self.ep_rewards, self.ep_envs) if e == env_name]
                env_landed = sum(1 for o, e in zip(self.ep_outcomes, self.ep_envs) if e == env_name and o == "landed")
                if env_rewards:
                    print("    {0}: n={1}  mean_rew={2:.2f}  landed={3}".format(
                        env_name, len(env_rewards), sum(env_rewards) / len(env_rewards), env_landed))

        print("=" * 60)

    def close(self):
        self.ep_file.close()
        self.upd_file.close()
        self.traj_file.close()
        print("Logger closed. Files saved.")
