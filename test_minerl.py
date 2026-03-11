import time
import logging
logging.basicConfig(level=logging.INFO)

print("=" * 50)
print("MineRL Install Test")
print("=" * 50)

# Step 1: Import check
print("\n[1/4] Testing imports...")
try:
    import gym
    import minerl
    print("  ✓ gym and minerl imported successfully")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Step 2: Environment registration check
print("\n[2/4] Checking registered MineRL environments...")
try:
    envs = [e.id for e in gym.envs.registry.values() if 'MineRL' in e.id]
    print(f"  ✓ Found {len(envs)} MineRL environments")
    print(f"  First 5: {envs[:5]}")
except Exception as e:
    print(f"  ✗ Registry check failed: {e}")

# Step 3: Make environment
print("\n[3/4] Creating MineRLNavigateDense-v0 environment...")
print("  (This will launch Minecraft — may take 30-60 seconds)")
try:
    env = gym.make('MineRLNavigateDense-v0')
    print("  ✓ Environment created")
except Exception as e:
    print(f"  ✗ gym.make failed: {e}")
    exit(1)

# Step 4: Reset / launch Minecraft
print("\n[4/4] Resetting environment (launches Minecraft)...")
print("  Waiting 5 seconds for Malmo to settle...")
time.sleep(5)
try:
    obs = env.reset()
    print(f"  ✓ Success! Observation shape: {obs['pov'].shape}")
    print("\n" + "=" * 50)
    print("  MineRL is working correctly!")
    print("=" * 50)
except Exception as e:
    print(f"  ✗ env.reset() failed: {e}")
finally:
    print("\nClosing environment...")
    env.close()
    print("Done.")
