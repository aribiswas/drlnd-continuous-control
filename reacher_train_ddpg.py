# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from agents import DDPGAgent
from matplotlib import pyplot as plt
import numpy as np
import collections
import torch

# DDPG hyperparameters
BUFFER_LENGTH = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
ALPHA_CRITIC = 1e-3
ALPHA_ACTOR = 1e-4
TAU = 0.001
UPDATE_FREQ = 10

# training options
MAX_EPISODES = 500      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages

# create environment
env = UnityEnvironment(file_name='Reacher20.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# observation and action specs
osize = 33
asize = 4

# create PPO agent
agent = DDPGAgent(osize, 
                 asize, 
                 seed=0,
                 buffer_length=BUFFER_LENGTH,
                 batch_size=BATCH_SIZE,
                 gamma=GAMMA,
                 tau=TAU,
                 update_freq=UPDATE_FREQ,
                 actor_LR=ALPHA_ACTOR,
                 critic_LR=ALPHA_CRITIC)

# score logs
reward_log = []
avg_log = []
avg_window = collections.deque(maxlen=AVG_WINDOW)

# verbosity
VERBOSE = True

solved = False

# Train the agent
for ep_count in range(1,MAX_EPISODES):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    
    ep_reward = 0
    
    while True:
        # sample action from the current policy
        actions = agent.get_action(states, train=True)
        
        # step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards 
        dones = env_info.local_done
        
        # step the agent
        agent.step(states,actions,rewards,next_states,dones)
        
        states = next_states
        ep_reward += np.sum(rewards)
        
        # terminate if done
        if np.any(dones):
            break
    
    # scale episode reward
    ep_reward /= 20
    
    # print training progress
    avg_window.append(ep_reward)
    avg_reward = np.mean(avg_window)
    avg_log.append(avg_reward)
    reward_log.append(ep_reward)
    if VERBOSE and (ep_count==1 or ep_count%50==0):
        print('Episode: {:4d} \tAverage Reward: {:6.2f} \tActor Loss: {:8.4f} \tCritic Loss: {:8.4f} \tNoise: {:6.4f}'.format(ep_count,avg_reward,agent.actor_loss_log[-1],agent.critic_loss_log[-1],agent.noise_log[-1]))
        
    # check if env is solved
    if not solved and avg_reward >= 30:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:6.2f}'.format(ep_count, avg_reward))
        solved = True

# save the policy
torch.save(agent.actor.state_dict(), 'checkpoint.pth')

# Close environment
env.close()

# plot score history
plt.ion()
fig1, ax1 = plt.subplots(1,1, figsize=(8,4), dpi=200)
ax1.set_title("Training Results")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Average Reward")
ax1.plot(avg_log)

fig2, axarr = plt.subplots(2,1, figsize=(6,4), dpi=200)
# plot loss
ax2 = axarr[0]
ax2.set_xlabel("Steps")
ax2.set_ylabel("Actor Loss")
ax2.plot(agent.actor_loss_log)

ax3 = axarr[1]
ax3.set_xlabel("Steps")
ax3.set_ylabel("Critic Loss")
ax3.plot(agent.critic_loss_log)

fig2.tight_layout(pad=1.0)
plt.show()
fig1.savefig('results_ddpg_scores.png',dpi=200)
fig2.savefig('results_ddpg_losses.png',dpi=200)





