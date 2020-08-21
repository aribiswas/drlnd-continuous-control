from unityagents import UnityEnvironment
from agents import DDPGAgent
from matplotlib import pyplot as plt
import numpy as np
import torch

# sim options
NUM_SIMS = 1

# create environment
env = UnityEnvironment(file_name='Reacher20.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# observation and action specs
osize = 33
asize = 4

# create PPO agent
agent = DDPGAgent(osize, asize, seed=0)

# load the weights from file
agent.actor.load_state_dict(torch.load('checkpoint.pth'))

# Sim the agent
for sim_count in range(NUM_SIMS):

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    
    ep_reward = 0
    
    while True:
        # sample action from the current policy
        actions = agent.get_action(states)
        
        # step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards 
        dones = env_info.local_done
        
        states = next_states
        ep_reward += np.sum(rewards)
        
        # terminate if done
        if np.any(dones):
            break
    
    # scale episode reward
    ep_reward /= 20
    
    # print training progress
    print('Sim: {:4d} \tCumulative Reward: {:6.2f}'.format(sim_count+1, ep_reward))

# Close environment
env.close()