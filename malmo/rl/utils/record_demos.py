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
import functools

# Force unbuffered stdout so logs appear in real time
print = functools.partial(print, flush=True)

PARKOUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARKOUR_ROOT)


from envs.env_client import EnvClient

# Import keyboard library for real-time key detection
try:
    import keyboard
except ImportError:
    print("ERROR: 'keyboard' package required. Install with: pip install keyboard")
    sys.exit(1)

# Import mouse library for right-click detection (bridging env)
try:
    import mouse
except ImportError:
    mouse = None  # only required for bridging env

# Environment registry (mirrors train.py) — only need configs for INPUT_SIZE
from training.configs.simple_jump_cfg     import SimpleJumpCFG
from training.configs.one_block_gap_cfg   import OneBlockGapCFG
from training.configs.three_block_gap_cfg import ThreeBlockGapCFG
from training.configs.bridging_cfg        import BridgingCFG

ENV_CONFIGS = {
    "simple_jump":     SimpleJumpCFG,
    "one_block_gap":   OneBlockGapCFG,
    "three_block_gap": ThreeBlockGapCFG,
    "bridging":        BridgingCFG,
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
    """Read held keys and translate to the best-matching parkour action index.

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


def translate_keys_to_action_bridging():
    """Read held keys and translate to bridging action index.

    Bridging action space (14 actions):
        0: move_forward       1: move_backward
        2: strafe_left        3: strafe_right
        4: look_down          5: look_up
        6: turn_left          7: turn_right
        8: sneak              9: place_block
       10: sneak_forward     11: sneak_backward
       12: sneak_place       13: no_op

    Keyboard mapping:
        W/S/A/D         = movement
        Arrows          = camera (look/turn)
        Shift           = sneak (hold/release)
        Right-click     = place block
        Shift+W         = sneak_forward
        Shift+S         = sneak_backward
        Shift+Rclick    = sneak_place

    Returns:
        Action index (int) or None if no keys are held.
    """
    w     = keyboard.is_pressed("w")
    s     = keyboard.is_pressed("s")
    a     = keyboard.is_pressed("a")
    d     = keyboard.is_pressed("d")
    shift = keyboard.is_pressed("shift")
    up    = keyboard.is_pressed("up")
    down  = keyboard.is_pressed("down")
    left  = keyboard.is_pressed("left")
    right = keyboard.is_pressed("right")
    place = mouse.is_pressed("right") if mouse else False

    # ── 2-key combos (most specific first) ─────────────────────────────────
    if shift and place:
        return 12  # sneak_place
    if shift and w:
        return 10  # sneak_forward
    if shift and s:
        return 11  # sneak_backward
    # Camera while sneaking — arrow takes priority over bare sneak since the
    # action space has no sneak+camera combo and camera intent is more valuable
    if shift and down:
        return 4   # look_down
    if shift and up:
        return 5   # look_up
    if shift and left:
        return 6   # turn_left
    if shift and right:
        return 7   # turn_right

    # ── 1-key actions ──────────────────────────────────────────────────────
    if place:
        return 9   # place_block
    if shift:
        return 8   # sneak (no other key held)
    if w:
        return 0   # move_forward
    if s:
        return 1   # move_backward
    if a:
        return 2   # strafe_left
    if d:
        return 3   # strafe_right
    if down:
        return 4   # look_down
    if up:
        return 5   # look_up
    if left:
        return 6   # turn_left
    if right:
        return 7   # turn_right

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
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode (default: 300 = 15s at 20Hz)")
    return parser.parse_args()


def record():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]

    # Select the correct key translator and action names for this env
    if args.env == "bridging":
        if mouse is None:
            print("ERROR: 'mouse' package required for bridging. Install with: pip install mouse")
            sys.exit(1)
        key_translator = translate_keys_to_action_bridging
    else:
        key_translator = translate_keys_to_action
    action_names = [a[0] for a in cfg.ACTIONS]

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
    print("Max steps:   ", "{0} (~{1}s)".format(args.max_steps,
                                                   int(args.max_steps * args.tick_rate)))
    print()
    if args.env == "bridging":
        print("Controls (bridging):")
        print("  W=forward  S=back  A=strafe_L  D=strafe_R")
        print("  Shift=sneak (hold)  Right-click=place_block")
        print("  Shift+W=sneak_forward  Shift+S=sneak_backward")
        print("  Shift+Rclick=sneak_place  Mouse=look/turn")
        print("  Enter=finish episode  Esc=save & quit")
    else:
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
            print("Episode {0} — recording...".format(episode_num))
            obs = env.reset(force_reset=True)
            steps = []
            done = False
            total_reward = 0.0
            step_count = 0
            info = None

            prev_blocks_placed = 0

            while not done and running:
                # Check for quit
                if keyboard.is_pressed("esc"):
                    running = False
                    break

                # Check for manual episode reset
                if keyboard.is_pressed("enter"):
                    break

                # Enforce max steps per episode
                if step_count >= args.max_steps:
                    print("  (max steps reached)")
                    break

                keyboard_action = key_translator()

                # Always step the env each tick (no_op if no key held) so we
                # can detect camera movement via obs pitch/yaw delta.
                step_action = keyboard_action if keyboard_action is not None else 13  # 13=no_op
                obs_prev = obs.copy()
                obs, reward, done, info = env.step(step_action)
                total_reward += reward
                step_count += 1

                actions_to_record = []

                # Bridging: blocks-placed override on keyboard action
                if args.env == "bridging" and info:
                    new_placed = info.get("blocks_placed", 0)
                    if new_placed > prev_blocks_placed:
                        if keyboard_action not in (9, 12):
                            keyboard_action = 12  # sneak_place
                        prev_blocks_placed = new_placed

                if keyboard_action is not None:
                    actions_to_record.append(keyboard_action)

                # Bridging: detect camera movement from obs pitch/yaw delta and
                # append as an additional action. Mouse controls camera so
                # keyboard polling alone misses all camera input.
                # Store the exact target angle so replay can drive closed-loop
                # to the correct angle rather than guessing action counts.
                if args.env == "bridging":
                    pitch_delta = obs[2] - obs_prev[2]  # positive = look down
                    yaw_delta   = obs[1] - obs_prev[1]  # positive = turn right
                    if yaw_delta > 1.0:
                        yaw_delta -= 2.0
                    elif yaw_delta < -1.0:
                        yaw_delta += 2.0
                    CAMERA_THRESH = 0.01  # ~0.9° pitch or ~1.8° yaw
                    if abs(pitch_delta) > CAMERA_THRESH or abs(yaw_delta) > CAMERA_THRESH:
                        if abs(pitch_delta) >= abs(yaw_delta):
                            camera_action = 4 if pitch_delta > 0 else 5  # look_down / look_up
                            camera_target = {"pitch_delta": float(abs(pitch_delta) * 90.0)}
                        else:
                            camera_action = 7 if yaw_delta > 0 else 6    # turn_right / turn_left
                            camera_target = {"yaw_delta": float(abs(yaw_delta) * 180.0)}
                        if camera_action not in actions_to_record:
                            actions_to_record.append((camera_action, camera_target))

                # Record all actions sequentially, all using obs_prev.
                # Camera steps carry a target_yaw or target_pitch for closed-loop replay.
                for item in actions_to_record:
                    if isinstance(item, tuple):
                        rec_action, extra = item
                    else:
                        rec_action, extra = item, {}
                    step_record = {"obs": obs_prev.tolist(), "action": int(rec_action)}
                    step_record.update(extra)
                    steps.append(step_record)
                    placed = " [PLACED #{0}]".format(prev_blocks_placed) if rec_action in (9, 12) and prev_blocks_placed > 0 else ""
                    print("  step:{0:>4} | action:{1:<18} | reward:{2:>7.2f}{3}".format(
                        step_count, action_names[rec_action], total_reward, placed))

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
