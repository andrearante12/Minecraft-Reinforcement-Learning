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
        self.ep_writer.writerow(["episode", "reward", "steps", "outcome", "timestamp"])

        # Update log — columns written dynamically on first call
        self.upd_path    = base + "_updates.csv"
        self.upd_file    = open(self.upd_path, "w", newline="")
        self.upd_writer  = csv.writer(self.upd_file)
        self._upd_header = False   # header written on first log_update call

        # Rolling window for console summary
        self.window        = 100
        self.ep_rewards    = deque(maxlen=self.window)
        self.ep_steps      = deque(maxlen=self.window)
        self.ep_outcomes   = deque(maxlen=self.window)
        self.episode_count = 0
        self.update_count  = 0
        self.start_time    = time.time()

        print("Logging to:")
        print("  Episodes: {0}".format(self.ep_path))
        print("  Updates:  {0}".format(self.upd_path))

    def log_episode(self, episode, reward, steps, outcome):
        self.ep_writer.writerow([episode, round(reward, 4), steps, outcome,
                                  time.strftime("%H:%M:%S")])
        self.ep_file.flush()
        self.ep_rewards.append(reward)
        self.ep_steps.append(steps)
        self.ep_outcomes.append(outcome)
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
        print("=" * 60)

    def close(self):
        self.ep_file.close()
        self.upd_file.close()
        print("Logger closed. Files saved.")
