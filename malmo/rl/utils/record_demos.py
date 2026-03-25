"""
utils/record_demos.py
---------------------
Record human demonstrations for behavioral cloning.

Connects to an env server via EnvClient and maps keyboard inputs to
actions. Uses standard Minecraft controls — held key combinations are
translated into our 15-action discrete space by translate_keys_to_action().

Usage:
    conda activate train_env
    python Malmo/rl/utils/record_demos.py --env simple_jump --port 10002
    python Malmo/rl/utils/record_demos.py --env simple_jump --port 10002 --output demos/my_demos.json

Keyboard mapping (standard Minecraft):
    W          = move forward        Ctrl+W       = sprint forward
    S          = move backward       Ctrl+W+Space = sprint jump
    A          = strafe left         W+Space      = jump forward
    D          = strafe right        Ctrl+W+A+Space = sprint jump left
    Space      = jump                Ctrl+W+D+Space = sprint jump right
    Up/Down    = look up/down
    Left/Right = turn left/right

Controls:
    Enter = finish current episode (reset)
    Esc   = stop recording and save
"""

import sys
import os
import json
import argparse
import time

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)

from envs.env_client import EnvClient

# Import keyboard library for real-time key detection
try:
    import keyboard
except ImportError:
    print("ERROR: 'keyboard' package required. Install with: pip install keyboard")
    sys.exit(1)

# Environment registry (mirrors train.py) — only need configs for INPUT_SIZE
from training.configs.simple_jump_cfg     import SimpleJumpCFG
from training.configs.one_block_gap_cfg   import OneBlockGapCFG
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG

ENV_CONFIGS = {
    "simple_jump":     SimpleJumpCFG,
    "one_block_gap":   OneBlockGapCFG,
    "three_block_gap": ThreeBlockGapCFG,
}

# Action indices matching base_cfg.py DEFAULT_ACTIONS
ACTION_NAMES = [
    "move_forward",       # 0
    "move_backward",      # 1
    "strafe_left",        # 2
    "strafe_right",       # 3
    "sprint_forward",     # 4
    "jump",               # 5
    "sprint_jump",        # 6
    "jump_forward",       # 7
    "sprint_jump_left",   # 8
    "sprint_jump_right",  # 9
    "look_down",          # 10
    "look_up",            # 11
    "turn_left",          # 12
    "turn_right",         # 13
    "no_op",              # 14
]


def translate_keys_to_action():
    """Read held keys and translate to the best-matching action index.

    Uses standard Minecraft controls. Combinations are checked from most
    specific (4 keys) to least specific (1 key) so the richest matching
    composite action is always selected.

    Returns:
        Action index (int) or None if no movement keys are held.
    """
    w     = keyboard.is_pressed("w")
    s     = keyboard.is_pressed("s")
    a     = keyboard.is_pressed("a")
    d     = keyboard.is_pressed("d")
    space = keyboard.is_pressed("space")
    ctrl  = keyboard.is_pressed("ctrl")
    up    = keyboard.is_pressed("up")
    down  = keyboard.is_pressed("down")
    left  = keyboard.is_pressed("left")
    right = keyboard.is_pressed("right")

    # ── 4-key combos (most specific first) ─────────────────────────────────
    if ctrl and w and a and space:
        return 8   # sprint_jump_left
    if ctrl and w and d and space:
        return 9   # sprint_jump_right

    # ── 3-key combos ───────────────────────────────────────────────────────
    if ctrl and w and space:
        return 6   # sprint_jump

    # ── 2-key combos ───────────────────────────────────────────────────────
    if ctrl and w:
        return 4   # sprint_forward
    if w and space:
        return 7   # jump_forward

    # ── 1-key actions ──────────────────────────────────────────────────────
    if w:
        return 0   # move_forward
    if s:
        return 1   # move_backward
    if a:
        return 2   # strafe_left
    if d:
        return 3   # strafe_right
    if space:
        return 5   # jump
    if down:
        return 10  # look_down
    if up:
        return 11  # look_up
    if left:
        return 12  # turn_left
    if right:
        return 13  # turn_right

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Record human demonstrations")
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_CONFIGS.keys()),
                        help="Environment to record in")
    parser.add_argument("--port", type=int, default=9999,
                        help="Env server TCP port (default: 9999)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: demos/<env>.json)")
    parser.add_argument("--tick-rate", type=float, default=0.05,
                        help="Seconds between ticks (default: 0.05 = 20Hz)")
    return parser.parse_args()


def record():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]

    output_path = args.output or os.path.join(PARKOUR_ROOT, "..", "demos",
                                               "{0}.json".format(args.env))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load existing demos if file exists (append mode)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
        print("Loaded {0} existing episodes from {1}".format(
            len(data["episodes"]), output_path))
    else:
        data = {"env": args.env, "episodes": []}

    env = EnvClient(cfg.INPUT_SIZE, port=args.port)

    print("=" * 60)
    print("Demo Recorder")
    print("=" * 60)
    print("Environment: ", args.env)
    print("Output:      ", output_path)
    print("Tick rate:   ", "{0}s ({1}Hz)".format(args.tick_rate,
                                                   int(1 / args.tick_rate)))
    print()
    print("Controls (standard Minecraft):")
    print("  W=forward  S=back  A=strafe_L  D=strafe_R")
    print("  Space=jump  Ctrl=sprint (hold with W)")
    print("  Ctrl+W+Space=sprint_jump  W+Space=jump_fwd")
    print("  Ctrl+W+A+Space=sprint_jump_L  Ctrl+W+D+Space=sprint_jump_R")
    print("  Arrows=look/turn")
    print("  Enter=finish episode  Esc=save & quit")
    print("=" * 60)
    print()

    episode_num = len(data["episodes"]) + 1
    running = True

    try:
        while running:
            print("Episode {0} — press any action key to start...".format(episode_num))
            obs = env.reset()
            steps = []
            done = False
            total_reward = 0.0
            step_count = 0
            info = None

            while not done and running:
                # Check for quit
                if keyboard.is_pressed("esc"):
                    running = False
                    break

                # Check for manual episode end
                if keyboard.is_pressed("enter"):
                    print("  (manual reset)")
                    time.sleep(0.3)  # debounce
                    break

                action = translate_keys_to_action()

                if action is not None:
                    # Record and execute
                    steps.append({
                        "obs": obs.tolist(),
                        "action": int(action),
                    })

                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    step_count += 1

                    # Real-time feedback
                    if step_count % 5 == 0:
                        print("  step:{0:>3} | action:{1:<18} | reward:{2:>6.2f}".format(
                            step_count, ACTION_NAMES[action], total_reward), end="\r")

                time.sleep(args.tick_rate)

            # Save episode if it has steps
            if steps:
                outcome = "manual_reset"
                if done and info:
                    outcome = info.get("outcome", "unknown")

                episode = {
                    "outcome": outcome,
                    "steps": steps,
                }
                data["episodes"].append(episode)
                print("\n  Episode {0}: {1} steps, reward={2:.2f}, outcome={3}".format(
                    episode_num, len(steps), total_reward, outcome))
                episode_num += 1

                # Auto-save after each episode
                with open(output_path, "w") as f:
                    json.dump(data, f)
                print("  (saved to {0})".format(output_path))
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        env.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(data, f)

    total_steps = sum(len(ep["steps"]) for ep in data["episodes"])
    print("\nRecording complete.")
    print("  Episodes: {0}".format(len(data["episodes"])))
    print("  Total steps: {0}".format(total_steps))
    print("  Saved to: {0}".format(output_path))


if __name__ == "__main__":
    record()
