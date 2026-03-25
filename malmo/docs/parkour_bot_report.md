# ParkourBot Architecture Report

A comprehensive overview of the system architecture, design decisions, and the reasoning behind them.

## 1. Problem Statement

Train an RL agent to perform parkour in Minecraft — jumping across gaps of increasing difficulty. The agent must learn to sprint, time jumps, and navigate 3D block geometry using only a compact observation vector, not raw pixels.

The core technical constraint: Microsoft Malmo's Python bindings require Python 3.7, while modern PyTorch requires Python 3.10+. These cannot coexist in the same process.

## 2. System Architecture

### 2.1 Two-Process Design

```
malmo conda env (Python 3.7)               train_env conda env (Python 3.10)
────────────────────────────                ─────────────────────────────────
env_server.py (TCP :9999)          ←→       train.py
  └── parkour_env.py                          ├── env_client.py
        └── MalmoPython + Minecraft JVM       ├── models/actor_critic.py
                                              ├── algos/ppo.py | dqn.py | bc.py
                                              ├── training/curriculum.py
                                              └── utils/logger.py
```

**Why two processes?** Malmo ships a pre-compiled `MalmoPython.pyd` binding locked to Python 3.7. PyTorch dropped 3.7 support years ago. Rather than trying to backport either (fragile and unmaintainable), we split the system at its natural boundary: the environment lives in one process, the learning algorithm in another. They communicate over TCP using a simple JSON protocol.

**Why TCP over alternatives?**
- **Shared memory / multiprocessing**: Requires the same Python version on both ends. Defeats the purpose.
- **Files / pipes**: Adds filesystem overhead, more error-prone for bidirectional communication.
- **TCP sockets**: Language-agnostic, version-agnostic, works across machines if needed. The protocol is trivial (`{"cmd": "step", "action": 3}` → `{"obs": [...], "reward": 0.5, "done": false, "info": {...}}`), adding negligible latency compared to Minecraft's tick time.

### 2.2 Communication Protocol

Messages are length-prefixed JSON over TCP:

```
[4-byte big-endian length][JSON payload]
```

Commands:
| Command | Payload | Response |
|---------|---------|----------|
| `reset` | `{}` | `{"obs": [...]}` |
| `step` | `{"action": int}` | `{"obs": [...], "reward": float, "done": bool, "info": {...}}` |
| `switch_env` | `{"env": "name"}` | `{"status": "ok"}` |
| `close` | `{}` | (connection closed) |

**Why length-prefixed?** TCP is a stream protocol — without framing, you can't tell where one message ends and the next begins. Length-prefixing is the simplest reliable framing scheme. We use 4-byte big-endian unsigned int (`struct.pack(">I")`) which supports messages up to 4 GB, far more than needed.

**Why JSON over protobuf/msgpack?** Observation vectors are ~159 floats. At this scale, serialization overhead is negligible compared to Minecraft's world tick (~50ms). JSON is human-readable for debugging and requires no schema compilation step.

## 3. Observation Space

**159-dimensional vector**, split into three semantic streams:

| Range | Name | Dims | Content |
|-------|------|------|---------|
| [0:6] | Proprioception | 6 | onGround, yaw, pitch, delta_y, delta_x, delta_z |
| [6:9] | Goal delta | 3 | goal_dx, goal_dy, goal_dz |
| [9:159] | Voxel grid | 150 | 5×5×6 grid of block types (0=air, 1=stone) |

### 3.1 Why Vector Observations Over Pixels?

- **Sample efficiency**: A 159-dim vector encodes exactly the information the agent needs. Pixel-based approaches (like VPT) require millions of frames and massive vision models to extract the same information.
- **Model size**: Our entire ActorCritic is ~50K parameters. VPT's vision backbone alone is orders of magnitude larger.
- **Training speed**: Vector observations enable training in hours on a single GPU, not days on a cluster.
- **Interpretability**: Every dimension has a known meaning, making debugging straightforward.

The tradeoff: the agent can't learn from visual cues that aren't in the vector (block textures, particles, etc.). For parkour, the included information is sufficient.

### 3.2 Proprioception Design

Velocity is inferred from position deltas (`pos_now - pos_prev`) rather than using Malmo's velocity fields. **Why?** Malmo's reported velocities can lag behind the actual physics state. Position differencing is always consistent with what the agent actually experienced.

Yaw and pitch are normalized to [-1, 1] (dividing by 180° and 90° respectively) to keep all proprioception values on a similar scale without requiring a separate normalization pass.

### 3.3 Voxel Grid Design

The grid covers x[-2:+2] × y[-1:+3] × z[-2:+3] = 5×5×6 = 150 blocks. This asymmetric extent was chosen deliberately:

- **X (width ±2)**: The agent needs to see blocks to its sides for strafing decisions, but parkour courses are narrow — 2 blocks of peripheral vision is enough.
- **Y (height -1 to +3)**: The agent needs to see the block under its feet (-1) and blocks above for jump clearance (+3). More below isn't useful — if you're that far down, you've already fallen.
- **Z (depth -2 to +3)**: Forward visibility (+3) is critical for planning jumps. Backward visibility (-2) provides context about where the agent came from.

Blocks are encoded as simple integers (air=0, stone=1). **Why not one-hot?** With only 2 block types in current environments, one-hot encoding would double the voxel dimensions for no benefit. The integer encoding extends naturally — adding a new block type just adds a new integer value without changing the vector size.

## 4. Action Space

**15 discrete composite actions**, each mapping to one or more simultaneous Malmo commands:

| Index | Action | Malmo Commands |
|-------|--------|----------------|
| 0 | move_forward | `move 1` |
| 1 | move_backward | `move -1` |
| 2 | strafe_left | `strafe -1` |
| 3 | strafe_right | `strafe 1` |
| 4 | sprint_forward | `sprint 1, move 1` |
| 5 | jump | `jump 1` |
| 6 | sprint_jump | `sprint 1, move 1, jump 1` |
| 7 | jump_forward | `move 1, jump 1` |
| 8 | sprint_jump_left | `sprint 1, move 1, strafe -1, jump 1` |
| 9 | sprint_jump_right | `sprint 1, move 1, strafe 1, jump 1` |
| 10 | look_down | `pitch 1` |
| 11 | look_up | `pitch -1` |
| 12 | turn_left | `turn -1` |
| 13 | turn_right | `turn 1` |
| 14 | no_op | (nothing) |

### 4.1 Why Composite Discrete Over Raw Continuous?

Minecraft's ContinuousMovementCommands API provides independent continuous axes (move, strafe, jump, sprint, pitch, turn). The agent could control each independently, but this creates a combinatorial explosion that makes exploration intractable.

**Composite discrete actions** pre-combine the common movement patterns a human Minecraft player would use. `sprint_jump` (index 6) is the single most important parkour action — it would require the agent to independently discover that `sprint=1 + move=1 + jump=1` must be activated simultaneously, which random exploration almost never finds.

The tradeoff: the agent can't perform arbitrary combinations (e.g., jump + turn simultaneously). For parkour, the pre-defined set covers all necessary movement patterns.

### 4.2 Action Execution Timing

Each action is a pulse: commands are sent, the agent waits `STEP_DURATION` seconds (0.15s, ~3 Minecraft ticks), then release commands are sent. This creates a fixed-frequency control loop that's simple and predictable.

**Why not hold actions across multiple ticks?** Variable-length actions add a temporal dimension to the policy that dramatically increases complexity. Fixed-duration pulses keep the MDP structure clean — one observation, one action, one transition.

## 5. Model Architecture

### 5.1 Multi-Stream ActorCritic

```
Proprioception (6) ──→ [Linear→LN→ReLU→Linear→LN→ReLU] (64)  ──┐
Goal delta (3)     ──→ [Linear→LN→ReLU→Linear→LN→ReLU] (64)  ──┼── concat (256)
Voxel grid (150)   ──→ [Linear→LN→ReLU→Linear→LN→ReLU] (128) ──┘
                                                                  │
                                                          ┌───────┴───────┐
                                                          │               │
                                              Actor head (256→15)  Critic head (256→1)
```

**Why three separate streams?** The three input groups have fundamentally different statistical properties:
- **Proprioception** is low-dimensional continuous data (velocities, angles).
- **Goal delta** is a 3D displacement vector with different scale and semantics.
- **Voxel grid** is high-dimensional sparse binary data.

Processing them independently before merging lets each stream learn appropriate feature transformations. A single monolithic network would need to disentangle these representations internally, wasting capacity.

### 5.2 Why LayerNorm Over BatchNorm?

LayerNorm normalizes across features within a single sample. BatchNorm normalizes across the batch dimension. For RL:
- **Batch sizes vary**: PPO minibatches can be small. BatchNorm's statistics become noisy with small batches.
- **Non-stationarity**: The data distribution shifts as the policy changes. BatchNorm's running statistics can lag behind. LayerNorm has no running statistics — it's always computed fresh.
- **Inference consistency**: LayerNorm behaves identically during training and inference. No train/eval mode switching needed.

### 5.3 Weight Initialization

Orthogonal initialization with gain √2 for hidden layers, following the PPO best practices from Andrychowicz et al. (2021). The actor's final layer uses a near-zero gain (0.01) to start with a near-uniform policy, encouraging initial exploration. The critic's final layer uses gain 1.0 for standard value prediction.

### 5.4 Model Size (~50K parameters)

The model is deliberately small. **Why?**
- The observation space is compact (159 dims). A larger model would overfit.
- Training runs on a single GPU in hours. There's no need for a model that requires distributed training.
- Inference must be fast — the control loop runs at ~7 Hz (0.15s step duration), and forward passes must be negligible.
- Small models are easier to debug, iterate on, and reason about.

## 6. Training Algorithms

### 6.1 PPO (Primary Algorithm)

PPO is the workhorse algorithm, chosen for several reasons:

- **Stability**: The clipped objective prevents catastrophically large policy updates, which is critical when training on a live Minecraft environment where bad policies can be hard to recover from.
- **Sample efficiency for on-policy**: PPO reuses each rollout buffer for multiple gradient steps (N_EPOCHS=4), squeezing more learning from each batch of experience.
- **Well-understood**: Extensive literature on hyperparameter tuning and failure modes.

**Key implementation decisions:**

**GAE (Generalized Advantage Estimation)**: Uses GAE (λ=0.95) instead of simple Monte Carlo returns. GAE provides a bias-variance tradeoff — high λ approaches Monte Carlo (low bias, high variance), low λ approaches TD(0) (high bias, low variance). λ=0.95 is close to Monte Carlo but smooths out the high-variance reward signals from stochastic Minecraft physics.

**Observation normalization**: Running mean/std normalization using Welford's online algorithm. Without this, the voxel grid (binary 0/1, 150 dims) would dominate the proprioception signals (small continuous values, 6 dims) in gradient magnitude. Normalization puts all features on a comparable scale.

**Reward normalization**: Standard deviation normalization only (no mean subtraction). Mean subtraction would shift the reward baseline and interfere with value function learning. Std normalization prevents reward scale from coupling with the learning rate.

**Clipped value function**: Both the policy and value function use clipping. Value clipping prevents the critic from making large jumps that destabilize advantage estimation.

**Linear decay schedules**: Both learning rate and entropy coefficient decay linearly from start to end values over the course of training. Early training benefits from high LR and entropy (fast exploration), while late training needs fine-grained optimization with less noise.

### 6.2 DQN (Alternative Algorithm)

Included as an off-policy alternative. DQN reuses the ActorCritic model by treating actor logits as Q-values — an unconventional choice made to avoid maintaining a separate network architecture.

**When DQN over PPO?** DQN's replay buffer gives better sample efficiency in data-scarce scenarios. However, for parkour, PPO's on-policy learning with environment interaction has proven more effective in practice.

### 6.3 Behavioral Cloning (Pre-training)

BC provides a supervised learning phase before RL. A human plays through the environment, recording (observation, action) pairs. The policy head is trained via cross-entropy loss on this data.

**Why BC before PPO?** Complex parkour sequences (multi-gap jumps) have enormous exploration spaces. Random exploration almost never discovers the right sprint-jump timing. BC bootstraps a reasonable policy that already knows the basic movement patterns, then PPO fine-tunes beyond human performance.

**BC → PPO transition**: BC checkpoints are loaded directly by PPO via `--checkpoint`. The policy weights transfer seamlessly. The critic head (untrained during BC) bootstraps from scratch — GAE estimates converge within a few PPO updates.

## 7. Reward Design

### 7.1 Reward Components

| Component | Value | Purpose |
|-----------|-------|---------|
| Success (landed) | +10.0 | Primary objective signal |
| Fell off | -5.0 | Penalize falling into the gap |
| Timeout | -5.0 | Penalize inaction / getting stuck |
| Step penalty | -0.01 | Encourage efficiency (fewer steps = less penalty) |
| Progress | +0.5 × Δdist | Reward moving toward the goal |
| Near miss | +2.0 | Partial credit for timing out close to the goal |

### 7.2 Design Decisions

**Asymmetric terminal rewards** (+10 success vs -5 fail): Success is rewarded more heavily than failure is penalized to prevent the agent from learning a risk-averse "stay still" policy. If fell and timeout penalties were too large, the optimal policy would be to never move (guaranteed -5 timeout, but no risk of -5 fell). The +10 carrot makes attempting jumps worth the risk.

**Progress reward**: Dense shaping based on distance-to-goal reduction. Without this, the agent would receive no learning signal until it accidentally lands on the target platform — which may never happen in the early stages of training. Progress reward provides a gradient the agent can follow before it ever succeeds.

**Proximity-scaled terminal penalties**: When enabled, fell/timeout penalties are scaled by how far the agent got (0 at spawn, 1 at goal). An agent that sprinted to the edge and mistimed the jump gets a smaller penalty than one that walked off the side at spawn. This prevents the pathological case where the agent learns that "getting close = bigger penalty" and becomes averse to approaching the gap.

**Near-miss bonus**: If the agent times out within 1.5 blocks of Z_SUCCESS, it gets +2.0 instead of -5.0 timeout. This specifically rewards "almost made it" attempts, encouraging the agent to keep trying jump strategies that come close rather than abandoning them.

### 7.3 Step Penalty Magnitude

The step penalty (-0.01) is deliberately tiny relative to terminal rewards. A 30-step episode accumulates only -0.3 penalty, while success gives +10.0. This means the agent prioritizes reaching the goal over speed. A larger step penalty would push the agent to attempt risky shortcuts.

## 8. Environment Design

### 8.1 Config-Driven Environments

All environments use the same `ParkourEnv` class. Behavioral differences (gap width, spawn position, success conditions) are entirely config-driven. Each environment is defined by:
1. An XML mission file (Malmo geometry definition)
2. A config class inheriting from `BaseCFG` (spawn, goal, thresholds, rewards)

**Why not separate env classes?** Every parkour environment shares the same observation building, reward computation, and action execution logic. Only the geometry and thresholds differ. Config inheritance avoids duplicating hundreds of lines of identical code across environments.

### 8.2 Current Environments

| Environment | Gap Width | Difficulty | Notes |
|-------------|-----------|------------|-------|
| simple_jump | 2 blocks | Easy | Basic sprint-jump, the "hello world" task |
| one_block_gap | 1 block | Easiest | Even a walk-jump can clear it |
| three_block_gap | 3 blocks | Hard | Requires precise sprint-jump timing |

### 8.3 Mission XML Structure

Each mission XML defines:
- **Flat world** with stone base, no terrain features
- **DrawingDecorator** blocks to create the runway and landing platform
- **Observation grid** (5×5×6) for the voxel observation
- **Time limit** (6 seconds) as a server-side safety net
- **ContinuousMovementCommands** for fine-grained control

**Why flat world + DrawingDecorator?** Malmo's `FlatWorldGenerator` creates a uniform base, then `DrawingDecorator` places specific blocks. This is deterministic — every reset produces the exact same geometry. Procedural generation would add variance that complicates learning and makes experiments harder to reproduce.

### 8.4 Fast Reset vs Force Reset

`ParkourEnv` maintains two XML variants: one with `forceReset="true"` and one with `forceReset="false"`. Force reset regenerates the entire world (slow, ~2s), while fast reset just teleports the agent back to spawn (~0.5s). Force reset is only used on the first episode after an environment switch. This 4x speedup in reset time accumulates to hours saved over thousands of training episodes.

## 9. Multi-Environment and Curriculum Training

### 9.1 Parallel Environments

Multiple env servers can run simultaneously, each connected to its own Minecraft client instance. The training script connects to all of them and collects transitions in parallel.

**Why parallel envs?** PPO's rollout buffer (N_STEPS=512) fills N times faster with N environments. More importantly, parallel environments provide diverse experience — different environments may be at different points in their episodes, reducing correlation between consecutive transitions and improving gradient estimates.

### 9.2 Curriculum Scheduling

Three modes for training across environments of increasing difficulty:

**Sequential**: Fixed episode counts per stage. Simple and predictable, but wastes time if the agent masters an easy stage quickly or struggles on a hard one.

**Weighted**: Round-robin with configurable weights. Good for mixed training where the agent should practice all difficulties simultaneously.

**Adaptive**: Performance-gated progression. The agent advances to the next stage when its rolling success rate hits a target threshold (e.g., 80% success on one_block_gap → move to simple_jump). This is the most efficient mode because it:
- Never wastes time on already-mastered stages
- Doesn't advance until the agent is actually ready
- Has min/max episode bounds to prevent getting stuck or rushing

**Why three modes?** Different training scenarios need different scheduling strategies. Ablation experiments use sequential (reproducible). Production training uses adaptive (efficient). Weighted is useful for maintaining performance on earlier stages while training harder ones.

## 10. Extensibility Architecture

### 10.1 Registry Pattern

Both `env_server.py` and `train.py` use registries — dictionaries mapping string names to (class, config) tuples:

```python
ENV_REGISTRY = {
    "simple_jump": (ParkourEnv, SimpleJumpCFG),
    ...
}

ALGO_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
    "bc":  BehavioralCloning,
}
```

Adding a new environment or algorithm requires zero changes to the training loop, logging, checkpointing, or any other infrastructure. Just implement the interface and add one registry entry.

**Why registries over auto-discovery?** Explicit registration is predictable — you can see exactly what's available by reading the registry. Auto-discovery (scanning directories, importing by convention) introduces subtle bugs when files are present but not ready for use, or when import order matters.

### 10.2 BaseAgent Contract

All algorithms inherit from `BaseAgent` and implement:
- `collect_step(env, obs)` → take one environment step
- `update(last_obs)` → run one learning update, return loss dict
- `select_action(obs, greedy)` → choose an action for evaluation
- `buffer_full()` → signal when enough data has been collected

The training loop calls these methods without knowing which algorithm is running. The logger adapts to whatever keys `update()` returns. Checkpointing uses `save()`/`load()` from the base class with optional `_extra_state()` hooks.

### 10.3 Config Inheritance

```
BaseCFG
  ├── SimpleJumpCFG
  ├── OneBlockGapCFG
  └── ThreeBlockGapCFG
```

`BaseCFG` defines all shared hyperparameters (learning rates, reward values, network sizes). Environment configs override only what's different (spawn position, gap width, success threshold). This means changing a hyperparameter in `BaseCFG` propagates to all environments automatically.

## 11. Logging and Checkpointing

### 11.1 Dual CSV Logging

Two CSV files per training run:
- **Episodes CSV**: episode number, reward, steps, outcome, env name, timestamp
- **Updates CSV**: update number + dynamic loss columns

**Why dynamic columns?** Different algorithms report different metrics. PPO returns `policy_loss, value_loss, entropy, lr, entropy_coef`. DQN returns `q_loss, epsilon, q_mean`. BC returns `bc_loss, accuracy`. The logger writes the header on the first `log_update()` call using whatever keys are present.

### 11.2 Checkpoint Strategy

Checkpoints save every 100 episodes and on KeyboardInterrupt. Each checkpoint contains:
- Model weights (`model_state`)
- Optimizer state (`optimizer_state`)
- Algorithm-specific state (e.g., PPO's running normalization statistics, DQN's epsilon)

**Cross-algorithm checkpoint loading**: A BC checkpoint can be loaded by PPO because they share the same model architecture and `BaseAgent.save()`/`load()` format. PPO ignores BC's optimizer state and creates its own. The policy weights transfer directly.

**Curriculum state sidecars**: Adaptive curriculum state is saved as a separate `*_curriculum.pt` file alongside the main checkpoint. This keeps the checkpoint format clean while allowing curriculum-aware resumption.

## 12. Training Pipeline Summary

```
                    ┌─────────────┐
                    │  Minecraft  │
                    │   Client    │
                    │  (JVM)      │
                    └──────┬──────┘
                           │ Malmo protocol
                    ┌──────┴──────┐
                    │ env_server  │  Python 3.7
                    │ parkour_env │
                    └──────┬──────┘
                           │ TCP/JSON
                    ┌──────┴──────┐
                    │  train.py   │  Python 3.10
                    │  ┌────────┐ │
                    │  │ Agent  │ │  PPO / DQN / BC
                    │  └───┬────┘ │
                    │  ┌───┴────┐ │
                    │  │ Model  │ │  ActorCritic (~50K params)
                    │  └────────┘ │
                    │  ┌────────┐ │
                    │  │ Logger │ │  CSV + console
                    │  └────────┘ │
                    └─────────────┘
```

The full training loop is ~100 lines in `train.py`. It handles multi-env collection, curriculum switching, logging, checkpointing, and graceful interrupt — all without knowing which algorithm or environment is running. This is possible because the abstractions (BaseAgent, config inheritance, registries) cleanly separate concerns.
