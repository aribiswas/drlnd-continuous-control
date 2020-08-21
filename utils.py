#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains utility functions for reinforcement learning algorithms.

Created: Aug 05, 2020
Revised: _

@author: abiswas
"""

import copy
import torch
import random
import numpy as np
from collections import namedtuple, deque
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


# --- For PPO ---

def discounted_rtg(rewards, gamma):
    """
    Compute the discounted rewards-to-go.
    
    Input: r = [ r0,
                 r1, 
                 ..., 
                 rT ]
    
    Ouput: rtg = [ r[0] + gamma * r[1] + gamma^2 * r[2] + ...,
                   r[1] + gamma * r[2] + gamma^2 * r[3] + ...,
                   ...,
                   r[T]]

    Parameters
    ----------
    rewards : numpy array | batchsize X 1
        Rewards from a trajectory.
    gamma : number
        Discount factor.
    device : char, optional
        'cpu' or 'gpu'. The default is 'cpu'.

    Returns
    -------
    rtg : numpy array | batchsize X 1
        Discounted rewards-to-go.

    """
    
    T = len(rewards)
    rtg = np.zeros((T,1))
    for i in range(T):
        rtg[i] = np.sum([gamma**k * rewards[k] for k in range(i,T)])
    return rtg


def gae(rewards, dones, state_values, next_state_values, gamma, lambdaa):
    """
    Compute Generalized Advantage Estimates (GAE) for a trajectory.
    
    r = [ r0,
          r1, 
          ..., 
          rT ]
    
    d = [ r[0] + gamma*Vnext[0] - V[0],
          r[1] + gamma*Vnext[1] - V[1],
          ...,
          r[T] + gamma*Vnext[T] - V[T] ]
    
    adv = [d[0] + (gamma*lambdaa)*d[1] + (gamma*lambdaa)^2*d[2] + ...,
           d[1] + (gamma*lambdaa)*d[2] + (gamma*lambdaa)^2*d[3] + ...,
           ...,
           d[T]]

    Parameters
    ----------
    rewards : numpy array | batchsize x 1
        Rewards from a trajectory.
    dones : numpy array | batchsize x 1
        Done signals from a trajectory.
    state_values : numpy array | batchsize x 1
        V(s).
    next_state_values : numpy array | batchsize x 1
        V(s+1).
    gamma : number
        Discount factor.
    lambdaa : number
        GAE factor.

    Returns
    -------
    adv : numpy array | batchsize X 1
        Advantage estimates.

    """
    
    T = len(rewards)
    x = gamma * lambdaa
    coeffs = np.array([x**k for k in range(T)]).reshape(T,1)
    deltas = rewards + gamma * next_state_values * (1-dones) - state_values
    adv = np.zeros((T,1))
    for i in range(T):
        adv[i] = np.sum([(x**k) * deltas[k] for k in range(i,T)])
    return adv


def clipped_ppo_loss(old_logps, new_logps, advantages, clip_factor):
    """
    Compute the clipped PPO loss.

    Parameters
    ----------
    old_logps : torch tensor | batchsize x 1
        Old log probabilities.
    new_logps : torch tensor | batchsize x 1
        New log probabilities.
    advantages : torch tensor | batchsize x 1
        Advantage estimates.
    clip_factor : number
        Clip factor for ratio clipping.
    device : char, optional
        'cpu' or 'gpu'. The default is 'cpu'.

    Returns
    -------
    loss: torch tensor
        Clipped loss.
    clipped_ratio: torch tensor
        Clipped ratio of new and old probabilities.

    """

    # likelihood ratio
    ratio = torch.exp(new_logps - old_logps)
    
    # clipped ratio
    clipped_ratio = torch.clamp(ratio, 1-clip_factor, 1+clip_factor)
    
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return loss, torch.mean(clipped_ratio)


class PPOMemory:
    
    def __init__(self, state_dim, act_dim, max_len):
        """
        Initialize a replay memory for storing:
            States
            Actions
            Rewards
            Next states
            Dones
            State values 
            Next state values
            Log probabilities 
            Advantage estimates
            Discounted rewards-to-go
        
        All items are stored as numpy arrays.

        Parameters
        ----------
        state_dim : number
            Dimension of states.
        act_dim : number
            Dimension of actions.
        max_len : number
            Capacity of memory.

        """
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        self.reset()
        
    def add(self, state, action, reward, next_state, done, state_value, next_state_value, logp):
        """
        Add experiences to replay memory.

        """
        self.last_idx += 1
        i = self.last_idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.state_values[i] = state_value
        self.next_state_values[i] = next_state_value
        self.logprobs[i] = logp
        
    def reset(self):
        """
        Reset the replay memory items to empty. Usually called after training
        is performed. 

        """
        self.states = np.empty((self.max_len,self.state_dim))
        self.actions = np.empty((self.max_len,self.act_dim))
        self.rewards = np.empty((self.max_len,1))
        self.next_states = np.empty((self.max_len,self.state_dim))
        self.dones = np.empty((self.max_len,1))
        self.state_values = np.empty((self.max_len,1))
        self.next_state_values = np.empty((self.max_len,1))
        self.logprobs = np.empty((self.max_len,1))
        self.advantages = np.empty((self.max_len,1))
        self.drtg = np.empty((self.max_len,1))
        self.last_idx = -1
        
    def get(self, device='cpu'):
        """
        Get all experienes from the replay memory and convert to torch tensors.

        """
        # convert to tensors
        states_batch = torch.from_numpy(self.states).float().to(device)
        actions_batch = torch.from_numpy(self.actions).float().to(device)
        rewards_batch = torch.from_numpy(self.rewards).float().to(device)
        next_states_batch = torch.from_numpy(self.next_states).float().to(device)
        dones_batch = torch.from_numpy(self.dones).float().to(device)
        state_values_batch = torch.from_numpy(self.state_values).float().to(device)
        next_state_values_batch = torch.from_numpy(self.next_state_values).float().to(device)
        advantages_batch = torch.from_numpy(self.advantages).float().to(device)
        drtg_batch = torch.from_numpy(self.drtg).float().to(device)
        logprobs_batch = torch.from_numpy(self.logprobs).float().to(device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, state_values_batch, next_state_values_batch, advantages_batch, drtg_batch, logprobs_batch
    
    def sample(self, batch_idxs, device='cpu'):
        """
        Get the experienes in batch_idxs and convert to torch tensors.

        """
        # convert to tensors
        states_batch = torch.from_numpy(self.states[batch_idxs]).float().to(device)
        actions_batch = torch.from_numpy(self.actions[batch_idxs]).float().to(device)
        rewards_batch = torch.from_numpy(self.rewards[batch_idxs]).float().to(device)
        next_states_batch = torch.from_numpy(self.next_states[batch_idxs]).float().to(device)
        dones_batch = torch.from_numpy(self.dones[batch_idxs]).float().to(device)
        state_values_batch = torch.from_numpy(self.state_values[batch_idxs]).float().to(device)
        next_state_values_batch = torch.from_numpy(self.next_state_values[batch_idxs]).float().to(device)
        advantages_batch = torch.from_numpy(self.advantages[batch_idxs]).float().to(device)
        drtg_batch = torch.from_numpy(self.drtg[batch_idxs]).float().to(device)
        logprobs_batch = torch.from_numpy(self.logprobs[batch_idxs]).float().to(device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, state_values_batch, next_state_values_batch, advantages_batch, drtg_batch, logprobs_batch


# ---- for DDPG ----

class GaussianNoise:
    
    def __init__(self, size, mean=0, std=0.1, stdmin=0.01, decay=1e-6):
        self.size = size
        self.mean = mean
        self.std = std
        self.stdmin = stdmin
        self.decay = decay
        self.step_count = 0
        
    def step(self):
        with torch.no_grad():
            mean = self.mean * torch.ones(self.size)
            cov = self.std * torch.eye(self.size)
            dist = MultivariateNormal(mean, cov)
            noise = dist.rsample(torch.Size([20,1])).squeeze(1).numpy()
            self.std = max(self.stdmin, self.std * (1-self.decay))
            self.step_count += 1
        return 0.25 * noise
    

class OUNoise:
    
    def __init__(self, size, mean=0, mac=0.15, var=0.1, varmin=0.01, decay=1e-6, seed=0):
        np.random.seed(seed)
        self.mean = mean * np.ones(size)
        self.mac = mac
        self.var = var
        self.varmin = varmin
        self.decay = decay
        self.x = np.zeros(size) #0.25 * np.random.rand(20,4)
        self.xprev = self.x
        self.step_count = 0
        
    def step(self):
        r = self.x.shape[0]
        c = self.x.shape[1]
        self.x = self.xprev + self.mac * (self.mean - self.xprev) + self.var * np.random.randn(r,c)
        self.xprev = self.x
        dvar = self.var * (1-self.decay)
        self.var = np.maximum(dvar, self.varmin)
        self.step_count += 1
        return self.x
    
    

# ExperienceReplay stores experiences in a circular buffer.
# Each experience is a tuple of (state,action,reward,next_state,done)
class ExperienceReplay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.rew_buf = []
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.rew_buf.append(reward)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        
        return len(self.memory)
    
    
class ExperienceBuffer:
    
    def __init__(self, state_dim, act_dim, max_len):
        """
        Initialize a replay memory for storing:
            States
            Actions
            Rewards
            Next states
            Dones
            State values 
            Next state values
            Log probabilities 
            Advantage estimates
            Discounted rewards-to-go
        
        All items are stored as numpy arrays.

        Parameters
        ----------
        state_dim : number
            Dimension of states.
        act_dim : number
            Dimension of actions.
        max_len : number
            Capacity of memory.

        """
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        
        self.states = np.empty((self.max_len,self.state_dim))
        self.actions = np.empty((self.max_len,self.act_dim))
        self.rewards = np.empty((self.max_len,1))
        self.next_states = np.empty((self.max_len,self.state_dim))
        self.dones = np.empty((self.max_len,1))
        
        self.last_idx = -1
        
        
    def add(self, state, action, reward, next_state, done):
        """
        Add experiences to replay memory.

        """
        beg = self.last_idx + 1
        end = beg + 20
        
        # check if buffer is full and append from front
        if beg > self.max_len or end > self.max_len:
            beg -= self.max_len
            end = beg + 20
            
        self.states[beg:end] = state
        self.actions[beg:end] = action
        self.rewards[beg:end] = np.array(reward).reshape(-1,1)
        self.next_states[beg:end] = next_state
        self.dones[beg:end] = np.array(done).astype(np.float64).reshape(-1,1)
            
        self.last_idx = end - 1
        
        
    def sample(self, batch_size, device='cpu'):
        """
        Get randomly sampled experiences.

        """
        # random indices
        batch_idxs = np.random.choice(self.last_idx+1, batch_size)
        
        # convert to tensors
        states_batch = torch.from_numpy(self.states[batch_idxs]).float().to(device)
        actions_batch = torch.from_numpy(self.actions[batch_idxs]).float().to(device)
        rewards_batch = torch.from_numpy(self.rewards[batch_idxs]).float().to(device)
        next_states_batch = torch.from_numpy(self.next_states[batch_idxs]).float().to(device)
        dones_batch = torch.from_numpy(self.dones[batch_idxs]).float().to(device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch


    def __len__(self):
        """Return the current size of internal memory."""
        
        return self.last_idx + 1



# ---- not currently used ----

def td_targets(next_state_values, rewards, gamma, device='cpu'):
    
    T = next_state_values.size()[0]
    gammas = torch.tensor([gamma**k for k in range(T)]).float().reshape([T,1]).to(device)
    target = rewards + gammas * next_state_values
    
    return target

def random_batch_indices(batch_size, data_size):
    num_batches = int(np.floor(data_size / batch_size))
    
    rand_idxs = np.arange(num_batches*batch_size)
    np.random.shuffle(rand_idxs)
    rand_idxs = rand_idxs.reshape((num_batches,batch_size))
    
    batch_idxs = []
    for i in range(num_batches):
        batch_idxs.append(rand_idxs[i])
    
    rem = data_size % batch_size
    if rem != 0:
        addl_idxs = np.arange(num_batches*batch_size+1, data_size)
        np.random.shuffle(addl_idxs)
        batch_idxs.append(addl_idxs)
        
    return batch_idxs
    
    
# class ExperienceBuffer:
    
#     def __init__(self, state_dim, act_dim, max_len):
#         """
#         Initialize a replay memory for storing:
#             States
#             Actions
#             Rewards
#             Next states
#             Dones
#             State values 
#             Next state values
#             Log probabilities 
#             Advantage estimates
#             Discounted rewards-to-go
        
#         All items are stored as numpy arrays.

#         Parameters
#         ----------
#         state_dim : number
#             Dimension of states.
#         act_dim : number
#             Dimension of actions.
#         max_len : number
#             Capacity of memory.

#         """
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.max_len = max_len
        
#         self.states = np.empty((self.max_len,self.state_dim))
#         self.actions = np.empty((self.max_len,self.act_dim))
#         self.rewards = np.empty((self.max_len,1))
#         self.next_states = np.empty((self.max_len,self.state_dim))
#         self.dones = np.empty((self.max_len,1))
        
#         self.last_idx = -1
        
        
#     def add(self, state, action, reward, next_state, done):
#         """
#         Add experiences to replay memory.

#         """
#         beg = self.last_idx + 1
#         end = beg + 20
        
#         if end <= self.max_len:
#             self.states[beg:end] = state
#             self.actions[beg:end] = action
#             self.rewards[beg:end] = np.array(reward).reshape(-1,1)
#             self.next_states[beg:end] = next_state
#             self.dones[beg:end] = np.array(done).astype(np.float64).reshape(-1,1)
#         else:
#             end = self.max_len
#             self.states[beg:end] = state[0:end-beg+1]
#             self.actions[beg:end] = action[0:end-beg+1]
#             self.rewards[beg:end] = np.array(reward[0:end-beg+1]).reshape(-1,1)
#             self.next_states[beg:end] = next_state[0:end-beg+1]
#             self.dones[beg:end] = np.array(done[0:end-beg+1]).astype(np.float64).reshape(-1,1)
            
#             self.states[0:20-(end-beg+1)] = state[end-beg+1:]
#             self.actions[0:20-(end-beg+1)] = action[end-beg+1:]
#             self.rewards[0:20-(end-beg+1)] = np.array(reward[end-beg+1:]).reshape(-1,1)
#             self.next_states[0:20-(end-beg+1)] = next_state[end-beg+1:]
#             self.dones[0:20-(end-beg+1)] = np.array(done[end-beg+1:]).astype(np.float64).reshape(-1,1)
#             end = 20-(end-beg+1)
            
#         self.last_idx = end - 1
        
        
#     def sample(self, batch_size, device='cpu'):
#         """
#         Get randomly sampled experiences.

#         """
#         # random indices
#         batch_idxs = np.random.choice(self.last_idx+1, batch_size)
        
#         # convert to tensors
#         states_batch = torch.from_numpy(self.states[batch_idxs]).float().to(device)
#         actions_batch = torch.from_numpy(self.actions[batch_idxs]).float().to(device)
#         rewards_batch = torch.from_numpy(self.rewards[batch_idxs]).float().to(device)
#         next_states_batch = torch.from_numpy(self.next_states[batch_idxs]).float().to(device)
#         dones_batch = torch.from_numpy(self.dones[batch_idxs]).float().to(device)
        
#         return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch


#     def __len__(self):
#         """Return the current size of internal memory."""
        
#         return self.last_idx + 1



# # ---- not currently used ----

# def td_targets(next_state_values, rewards, gamma, device='cpu'):
    
#     T = next_state_values.size()[0]
#     gammas = torch.tensor([gamma**k for k in range(T)]).float().reshape([T,1]).to(device)
#     target = rewards + gammas * next_state_values
    
#     return target

# def random_batch_indices(batch_size, data_size):
#     num_batches = int(np.floor(data_size / batch_size))
    
#     rand_idxs = np.arange(num_batches*batch_size)
#     np.random.shuffle(rand_idxs)
#     rand_idxs = rand_idxs.reshape((num_batches,batch_size))
    
#     batch_idxs = []
#     for i in range(num_batches):
#         batch_idxs.append(rand_idxs[i])
    
#     rem = data_size % batch_size
#     if rem != 0:
#         addl_idxs = np.arange(num_batches*batch_size+1, data_size)
#         np.random.shuffle(addl_idxs)
#         batch_idxs.append(addl_idxs)
        
#     return batch_idxs