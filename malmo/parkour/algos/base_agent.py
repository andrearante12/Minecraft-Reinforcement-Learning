"""
algos/base_agent.py
-------------------
Abstract base class for all RL agents.

Every algorithm (PPO, DQN, SAC, etc.) must inherit from BaseAgent
and implement all abstract methods. This guarantees a consistent
interface so agents can be swapped without changing the training loop.

save() and load() are implemented here since they follow the same
pattern across all algorithms. Override _extra_state() to save any
algorithm-specific state beyond model and optimizer weights.

To implement a new algorithm:

    from algos.base_agent import BaseAgent

    class MyAgent(BaseAgent):
        def __init__(self, model, cfg):
            ...
            # must assign self.model and self.optimizer

        def collect_step(self, env, obs): ...
        def update(self, last_obs): ...
        def select_action(self, obs, greedy=False): ...
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseAgent(ABC):

    @abstractmethod
    def __init__(self, model, cfg):
        """
        Initialize the agent.
        Must assign self.model, self.optimizer, and self.device.
        """
        pass

    @abstractmethod
    def collect_step(self, env, obs: np.ndarray):
        """
        Take one step in the environment and store the transition.

        Returns:
            next_obs: np.ndarray
            reward:   float
            done:     bool
            info:     dict
        """
        pass

    @abstractmethod
    def update(self, last_obs: np.ndarray) -> dict:
        """
        Run one update step using collected experience.

        Returns:
            dict of loss metrics, e.g.:
            {"policy_loss": 0.05, "value_loss": 0.3, "entropy": 1.2}
        """
        pass

    @abstractmethod
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Select an action for the given observation.

        Args:
            greedy: if True, no exploration — used during evaluation.

        Returns:
            action index (int)
        """
        pass

    def buffer_full(self) -> bool:
        """
        Returns True when enough experience has been collected to update.
        Override for on-policy algorithms that use a rollout buffer.
        Default False = update every step (e.g. DQN).
        """
        return False

    # ── Checkpointing ──────────────────────────────────────────────────────────
    # Concrete implementations shared across all algorithms.
    # Override _extra_state() to save algorithm-specific state.

    def _extra_state(self) -> dict:
        """
        Override to include algorithm-specific state in the checkpoint.
        Example — DQN might return {"epsilon": self.epsilon}
        """
        return {}

    def _load_extra_state(self, state: dict):
        """Override to restore algorithm-specific state from checkpoint."""
        pass

    def save(self, path: str):
        """Save model, optimizer, and any extra algorithm state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            **self._extra_state(),
        }, path)
        print("Checkpoint saved:", path)

    def load(self, path: str):
        """Load model, optimizer, and any extra algorithm state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._load_extra_state(ckpt)
        print("Checkpoint loaded:", path)