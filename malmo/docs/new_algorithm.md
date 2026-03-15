# Adding New RL Algorithms

All algorithms inherit from `BaseAgent` (`algos/base_agent.py`) and must implement four methods. `save()` and `load()` are provided by `BaseAgent` and do not need to be reimplemented unless the algorithm has extra state to persist (e.g. epsilon for DQN).

## Required Methods

| Method | Description |
|--------|-------------|
| `__init__(model, cfg)` | Initialize the algorithm. Must assign `self.model`, `self.optimizer`, and `self.device`. |
| `collect_step(env, obs)` | Take one step, store the transition, return `(next_obs, reward, done, info)`. |
| `update(last_obs)` | Run one update using collected experience. Return a dict of loss metrics. |
| `select_action(obs, greedy)` | Return an action index. `greedy=True` is used during evaluation. |

## Optional Methods

| Method | Description |
|--------|-------------|
| `buffer_full()` | Return `True` when ready to update. Default is `False` (update every step). Override for on-policy algorithms like PPO that collect a full rollout before updating. |
| `_extra_state()` | Return a dict of any extra state to include in the checkpoint (e.g. `{"epsilon": self.epsilon}`). |
| `_load_extra_state(state)` | Restore extra state from a loaded checkpoint dict. |

## Steps to Add a New Algorithm

**1. Create `algos/my_algo.py` and inherit from `BaseAgent`:**

```python
import torch
import torch.optim as optim
from algos.base_agent import BaseAgent

class MyAlgo(BaseAgent):
    def __init__(self, model, cfg):
        self.model     = model
        self.cfg       = cfg
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    def collect_step(self, env, obs):
        action = self.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        # store transition...
        return next_obs, reward, done, info

    def update(self, last_obs=None):
        # compute loss, backprop...
        return {"my_loss": loss.item()}

    def select_action(self, obs, greedy=False):
        # return action index...
        pass
```

**2. Add any algorithm-specific hyperparameters to `training/config.py`:**

```python
# ── MyAlgo ───────────────────────────────────────────────────────────────────
MY_ALGO_PARAM_1 = 0.99
MY_ALGO_PARAM_2 = 1000
```

**3. Register the algorithm in `training/train_simple_jump.py`:**

```python
from algos.my_algo import MyAlgo

ALGO_REGISTRY = {
    "ppo":     PPO,
    "dqn":     DQN,
    "my_algo": MyAlgo,   # add this line
}
```

**4. Run training with the new algorithm:**

```powershell
python Malmo/rl/training/train_simple_jump.py --algo my_algo
```

The training loop, environment, logging, and checkpointing all work automatically — the `update()` return dict keys are logged dynamically so no changes to `logger.py` are needed either.

## How Logging Works

`logger.py` writes whatever keys `update()` returns directly to CSV. The header is set on the first call based on the keys in the returned dict. For example:

- PPO returns `{"policy_loss": ..., "value_loss": ..., "entropy": ...}` → CSV columns: `update, policy_loss, value_loss, entropy`
- DQN returns `{"q_loss": ..., "epsilon": ..., "q_mean": ...}` → CSV columns: `update, q_loss, epsilon, q_mean`
- Your algorithm returns `{"my_loss": ...}` → CSV columns: `update, my_loss`
