# Trying New Model Architectures

The model architecture is fully decoupled from the training loop and algorithms. PPO and DQN interact with the model through exactly three methods — as long as your new architecture implements these, it can be swapped in by changing a single import line.

## Required Methods

Any model must implement these three methods with the same signatures:

| Method | Input | Output |
|--------|-------|--------|
| `get_distribution(obs)` | `(1, INPUT_SIZE)` tensor | `Categorical` distribution |
| `get_value(obs)` | `(1, INPUT_SIZE)` tensor | `(1, 1)` scalar tensor |
| `evaluate_actions(obs, actions)` | `(batch, INPUT_SIZE)`, `(batch,)` | `log_probs (batch,)`, `values (batch,)`, `entropy (scalar)` |

## Constructor

The default `ActorCritic` (in `models/actor_critic.py`) takes a single `cfg` object. Required config attributes:

- `PROPRIOCEPTION_SIZE` — size of proprioception slice (default: 6)
- `GOAL_DELTA_SIZE` — size of goal delta slice (default: 3)
- `GRID_SIZE` — size of voxel grid slice (default: 120)
- `N_ACTIONS` — number of discrete actions
- `PROPRIO_HIDDEN` — hidden width for proprioception stream (default: 64)
- `GOAL_HIDDEN` — hidden width for goal stream (default: 64)
- `VOXEL_HIDDEN` — hidden width for voxel stream (default: 128)
- `HEAD_HIDDEN` — hidden width for actor/critic heads (default: 256)

The old `models/mlp.py` is kept as a fallback for loading old checkpoints. It uses `ActorCritic(input_size, hidden_size, n_actions)`.

## Steps to Add a New Architecture

**1. Create a new file in `models/`**

For example, a deeper MLP (`models/mlp_deep.py`):

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_size  = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        n_actions   = cfg.N_ACTIONS

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),  # extra layer
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head  = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        features = self.shared(obs)
        return self.policy_head(features), self.value_head(features)

    def get_distribution(self, obs):
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy().mean()
```

**2. Update the import in `training/train.py`**

```python
# Before
from models.actor_critic import ActorCritic

# After
from models.mlp_deep import ActorCritic
```

That's it — the training loop, PPO update, DQN update, logging, and checkpointing all work without any other changes.

## Observation Vector

All models receive the same 129-dimensional input. See [Observation Vector](./observation_vector.md) for the full breakdown of what each index represents.
