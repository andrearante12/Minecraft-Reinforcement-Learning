import json, sys
sys.path.insert(0, 'Malmo/rl')
with open('Malmo/demos/bridging_advanced.json') as f:
    d = json.load(f)
first_obs = d['episodes'][0]['steps'][0]['obs']
print('Demo obs size:', len(first_obs))
from training.configs.bridging_cfg import BridgingCFG
print('Config INPUT_SIZE:', BridgingCFG.INPUT_SIZE)
print('Match:', len(first_obs) == BridgingCFG.INPUT_SIZE)
actions = [s['action'] for ep in d['episodes'] for s in ep['steps']]
print('Total demo steps:', len(actions))
print('Place block (action 10) count:', actions.count(10))
print('Place block %:', round(actions.count(10)/len(actions)*100, 1))
