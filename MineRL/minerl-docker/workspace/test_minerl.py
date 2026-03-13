import gym
import minerl
import logging

logging.basicConfig(level=logging.INFO)
print("Initializing environment...")
env = gym.make('MineRLNavigateDense-v0')

print("Resetting... (This triggers the Minecraft build)")
obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if i % 10 == 0:
        print(f"Step {i}...")

print("Success!")
env.close()
