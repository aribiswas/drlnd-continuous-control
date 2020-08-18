# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from agents import PPOAgent
from matplotlib import pyplot as plt
import numpy as np
import collections

# PPO hyperparameters 
HORIZON = 1001
BATCH_SIZE = 256
GAMMA = 0.9995
LAMBDA = 0.95
ALPHA_CRITIC = 1e-3
ALPHA_ACTOR = 1e-3
CLIP_FACTOR = 0.2
ENTROPY_FACTOR = 0.01

# training options
MAX_EPISODES = 500      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages

# create environment
env = UnityEnvironment(file_name='Reacher.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# observation and action specs
osize = 33
asize = 4

# create PPO agent
agent = PPOAgent(osize, 
                 asize, 
                 horizon=HORIZON,
                 batch_size=BATCH_SIZE,
                 discount_factor=GAMMA, 
                 gae_factor=LAMBDA, 
                 actor_LR=ALPHA_ACTOR,
                 critic_LR=ALPHA_CRITIC, 
                 clip_factor=CLIP_FACTOR,
                 entropy_factor=ENTROPY_FACTOR)

# log scores
reward_log = []
avg_log = []
avg_window = collections.deque(maxlen=AVG_WINDOW)

# verbosity
VERBOSE = True

# Train the agent
for ep_count in range(1,MAX_EPISODES):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    
    ep_reward = 0
    
    while True:
        # sample action from the current policy
        action = agent.get_action(state)
        
        # step the environment
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0] 
        done = env_info.local_done[0]
        
        # step the agent
        agent.step(state,action,reward,next_state,done)
        
        state = next_state
        ep_reward += reward
        
        # terminate if done
        if done:
            break
    
    # print training progress
    avg_window.append(ep_reward)
    avg_reward = np.mean(avg_window)
    avg_log.append(avg_reward)
    reward_log.append(ep_reward)
    if VERBOSE and (ep_count==1 or ep_count%1==0):
        print('Episode: {:4d} \tAverage Reward: {:4.2f} \tActor Loss: {:8.4f} \tPPO Ratio: {:6.4f} \tCritic Loss: {:8.4f}'.format(ep_count,avg_reward,agent.actor_loss_log[-1],agent.ratio_log[-1],agent.critic_loss_log[-1]))
        
    # check if env is solved
    if avg_reward >= 30:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:4.2f}'.format(ep_count, avg_reward))
        torch.save(actor.state_dict(), 'checkpoint.pth')
        break

# Close environment
env.close()

# plot score history
plt.ion()
fig, axarr = plt.subplots(3,1, figsize=(4,4), dpi=200)
ax1 = axarr[0]
ax1.set_title("Training Results")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Average Reward")
ax1.plot(avg_log)

# plot loss
ax2 = axarr[1]
ax2.set_xlabel("Steps")
ax2.set_ylabel("Actor Loss")
ax2.plot(agent.actor_loss_log)

ax3 = axarr[2]
ax3.set_xlabel("Steps")
ax3.set_ylabel("Critic Loss")
ax3.plot(agent.critic_loss_log)

fig.tight_layout(pad=1.0)
plt.show()
fig.savefig('results.png',dpi=200)
    
