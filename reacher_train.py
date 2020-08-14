# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
from model import StochasticActor, Critic
from agents import PPOAgent
import numpy as np
import collections

# PPO hyperparameters
EPOCHS = 3
HORIZON = 512
BATCHSIZE = 128
GAMMA = 0.99
LAMBDA = 0.95
ALPHA = 0.001
EPSILON = 0.2
BETA = 0.01

# training options
MAX_EPISODES = 5000      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages
MAX_STEPS_PER_EPISODE = 1000    # Maximum agent steps per episode

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

# create actor and critic neural networks
actor = StochasticActor(osize, asize, seed=0)
critic = Critic(osize, seed=0)

# create PPO agent
agent = PPOAgent(actor, 
                 critic, 
                 horizon=HORIZON, 
                 batch_size=BATCHSIZE,
                 epochs=EPOCHS,
                 discount_factor=GAMMA, 
                 gae_factor=LAMBDA, 
                 actor_LR=ALPHA,
                 critic_LR=ALPHA, 
                 clip_factor=EPSILON,
                 entropy_factor=BETA)

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
    
    # DEBUG
    #print('Episode={:d}\n'.format(ep_count))
    #t_count = 0
    
    #for t in range(1,MAX_STEPS_PER_EPISODE):
    while True:
        # sample action from the current policy
        action = actor.get_action(state)
        
        # step the environment
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0] 
        done = env_info.local_done[0]
        
        # step the agent
        agent.step(state,action,reward,next_state,done)
        
        state = next_state
        ep_reward += reward
        
        #t_count += 1
        
        # terminate if done
        if done:
            # DEBUG
            #print('Terminated at={:d} steps\n'.format(t_count))
            break
    
    # print training progress
    avg_window.append(ep_reward)
    avg_reward = np.mean(avg_window)
    avg_log.append(avg_reward)
    reward_log.append(ep_reward)
    if VERBOSE and (ep_count==1 or ep_count%1==0):
        print('Episode: {:4d} \tAverage Reward: {:4.2f} \tActor Loss: {:6.4f} \tCritic Loss: {:6.4f}'.format(ep_count,avg_reward,agent.actor_loss_log[-1],agent.critic_loss_log[-1]))
    
    # check if env is solved
    if avg_reward >= 30:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.2f}'.format(ep_count, avg_reward))
        torch.save(agent.Q.state_dict(), 'checkpoint.pth')
        break

# Close environment
env.close()

# plot score history
plt.ion()
fig, axarr = plt.subplots(2,1, figsize=(4,4), dpi=200)
ax1 = axarr[0]
ax1.set_title("Training Results")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Average Reward")
ax1.set_xlim([0, ep_count+20])
ax1.set_ylim([0, 20])
ax1.plot(range(1,ep_count+1),avg_log)

# plot loss
ax2 = axarr[1]
ax2.set_xlabel("Steps")
ax2.set_ylabel("Loss")
ax2.set_xlim([0, agent.stepcount+20])
ax2.plot(range(agent.minibatchsize,agent.stepcount),agent.loss_log)

fig.tight_layout(pad=1.0)
plt.show()
fig.savefig('results.png',dpi=200)
    
