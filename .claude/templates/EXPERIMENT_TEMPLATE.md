# MalmoRL Experiment Request

> **How to use this form:**
> 1. Copy this file to the repo root and rename it (e.g. `my_experiment.md`).
> 2. Fill in the fields below in plain English. Leave a field blank, write `default`, or write `n/a` to skip it.
> 3. Hand the filled form to Claude with a phrase like *"run experiment-orchestrator on `my_experiment.md`"*.
> 4. The orchestrator will parse the form, ask clarifying questions for anything ambiguous, show you a dispatch plan, and (with your approval) invoke the right specialist sub-agents to scaffold everything.

---

## 1. Experiment name
<!-- snake_case, e.g. four_block_gap_sac. This becomes the env name, config filename, and registry key. -->



## 2. Environment

**Type:**
<!-- One of:
     "parkour variant"     — geometry-only change, reuses ParkourEnv
     "new task type"       — fundamentally different gameplay, needs a new env class
     "use existing: <name>" — train on an existing env (e.g. "bridging", "simple_jump")
-->


**Description:**
<!-- One paragraph: what does the world look like? Where does the agent spawn? Where is the goal? Any obstacles? -->


**Geometry details (parkour variants only):**
<!-- Gap widths, platform heights, lateral offsets, multi-jump layout, etc. E.g.: "4-block gap at z=4..7, landing platform at z=8..20, starting platform at z=-20..3, all at y=45 stone." -->



## 3. Goals & success conditions
<!-- What counts as success? What counts as failure? Are there time limits or step caps?
     Examples: "Reach the goal block within 200 steps. Fall below y=43 = fail."
               "Place blocks to cross the gap. Reach z>=15 to succeed. Run out of inventory = fail." -->



## 4. Rewards

**Reward shape:**
<!-- One of:
     "default for this env type"  — use the standard reward terms inherited from BaseCFG
     "describe terms"             — list each reward signal in plain English below
-->


**Specific values (optional):**
<!-- e.g.:
     +10 on reaching the goal
     -5 on falling below y=43
     -0.01 per step
     +0.5 per unit of progress toward the goal
-->



## 5. Algorithm

**Name:**
<!-- An existing algorithm ("ppo", "dqn", "bc") OR a new algorithm to scaffold (e.g. "sac", "a2c"). -->


**Family (only if new):**
<!-- One of: "on-policy" (PPO-like), "off-policy" (DQN-like), "imitation" (BC-like). -->


**Hyperparameters to override (optional):**
<!-- e.g. "learning rate 5e-5, batch size 128, gamma 0.995". Leave blank for defaults. -->



## 6. Model architecture

**Architecture:**
<!-- One of:
     "default"                                    — use the existing multi-stream ActorCritic
     "describe a custom architecture"             — e.g. "deeper MLP with 3 hidden layers of width 256"
                                                  — e.g. "CNN over the voxel grid before fusion"
                                                  — e.g. "recurrent (GRU) head"
-->



## 7. Launch options

**Demos / BC pre-training:**
<!-- Path to a demo JSON (e.g. "Malmo/demos/my_demos.json"), or "no". -->


**Resume from checkpoint:**
<!-- Path to a .pt or .zip checkpoint, or "no". -->


**Parallel envs:**
<!-- 1, 2, 4, 8, etc. Default 1. -->


**Auto-launch after scaffolding?**
<!-- "yes" — orchestrator prints the 3-terminal commands when done (does not launch automatically).
     "no"  — orchestrator stops after scaffolding. -->



## 8. Notes for Claude (optional)
<!-- Anything else — known constraints, things you've already tried, what you want to learn from this experiment, etc. -->
