"""
models/mlp.py
-------------
Actor-Critic network used by PPO.
 
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, n_actions):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head  = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

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
        """Returns log_probs, values, entropy for a batch — used in PPO update."""
        logits, values = self.forward(obs)
        dist           = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy().mean()


if __name__ == "__main__":
    from training.config import INPUT_SIZE, HIDDEN_SIZE, N_ACTIONS

    model = ActorCritic(INPUT_SIZE, HIDDEN_SIZE, N_ACTIONS)
    print(model)

    dummy_obs        = torch.randn(4, INPUT_SIZE)
    logits, value    = model(dummy_obs)
    print("Input: ", dummy_obs.shape)
    print("Logits:", logits.shape)
    print("Value: ", value.shape)
    print("Entropy:", model.get_distribution(dummy_obs).entropy().mean().item())
