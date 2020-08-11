#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains utility functions for reinforcement learning algorithms.

Created: Aug 05, 2020
Revised: _

@author: abiswas
"""

import torch
import numpy as np

def compute_td_targets(next_state_values, rewards, gamma, device='cpu'):
    
    T = next_state_values.size()[0]
    
    gammas = torch.tensor([gamma**k for k in range(T)]).float().reshape([T,1]).to(device)
    target = rewards + gammas * next_state_values
    
    return target


def advantage_function(states, 
                       rewards, 
                       state_values, 
                       next_state_values, 
                       horizon, 
                       gamma, 
                       lambdaa):

    x = gamma * lambdaa
    coeffs = np.array([x**k for k in range(horizon)]).reshape(horizon,1)
    targets = rewards + gamma * next_state_values
    deltas = targets - state_values
    advantages = coeffs * deltas
    
    return advantages


def clipped_ppo_loss(old_logps, new_logps, advantages, clip_factor, device='cpu'):

    # likelihood ratio
    ratio = torch.exp(new_logps - old_logps)
    
    # clipped loss
    loss = torch.clamp(ratio * advantages, 1-clip_factor, 1+clip_factor).to(device)
    
    #return -loss.mean(dim=1,keepdim=True)
    
    return -loss.mean()


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


class PPOMemory:
    
    def __init__(self, state_dim, act_dim, max_len):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        self.reset()
        
    def add(self, state, action, reward, next_state, done, logp):
        self.last_idx += 1
        i = self.last_idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.logprobs[i] = logp
        
    def reset(self):
        self.states = np.empty((self.max_len,self.state_dim))
        self.actions = np.empty((self.max_len,self.act_dim))
        self.rewards = np.empty((self.max_len,1))
        self.next_states = np.empty((self.max_len,self.state_dim))
        self.dones = np.empty((self.max_len,1))
        self.logprobs = np.empty((self.max_len,self.act_dim))
        self.advantages = np.empty((self.max_len,1))
        self.last_idx = -1
    
    def sample(self, batch_idxs, device='cpu'):
        # convert to tensors
        states_batch = torch.from_numpy(self.states[batch_idxs]).float().to(device)
        actions_batch = torch.from_numpy(self.actions[batch_idxs]).float().to(device)
        rewards_batch = torch.from_numpy(self.rewards[batch_idxs]).float().to(device)
        next_states_batch = torch.from_numpy(self.next_states[batch_idxs]).float().to(device)
        advantages_batch = torch.from_numpy(self.advantages[batch_idxs]).float().to(device)
        logprobs_batch = torch.from_numpy(self.logprobs[batch_idxs]).float().to(device)
        
        # states_batch = self.states[batch_idxs]
        # actions_batch = self.actions[batch_idxs]
        # rewards_batch = self.rewards[batch_idxs]
        # next_states_batch = self.next_states[batch_idxs]
        # advantages_batch = self.advantages[batch_idxs]
        # logprobs_batch = self.logprobs[batch_idxs]
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, advantages_batch, logprobs_batch
        

# class PPOMemory:
    
#     def __init__(self, state_dim, act_dim, max_len):
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.max_len = max_len
#         self.last_idx = -1
        
#         self.reset()
        
#     def add(self, state, action, reward, next_state, done, logp):
#         self.last_idx += 1
#         self.states[self.last_idx] = state
#         self.actions[self.last_idx] = action
#         self.rewards[self.last_idx] = float(reward)
#         self.next_states[self.last_idx] = next_state
#         self.dones[self.last_idx] = float(done)
#         self.logprobs[self.last_idx] = logp
        
#     def reset(self):
#         self.states = np.empty([self.max_len,self.state_dim])
#         self.actions = np.empty([self.max_len,self.act_dim])
#         self.rewards = np.empty([self.max_len,1])
#         self.next_states = np.empty([self.max_len,self.state_dim])
#         self.dones = np.empty([self.max_len,1])
#         self.logprobs = np.empty([self.max_len,self.act_dim])
#         self.advantages = np.empty([self.max_len,1])
#         self.last_idx = -1
    
#     def sample(self, batch_idxs):
#         states_batch = torch.from_numpy(self.states[batch_idxs]).float().to(device)
#         actions_batch = torch.from_numpy(self.actions[batch_idxs]).float().to(device)
#         rewards_batch = torch.from_numpy(self.rewards[batch_idxs]).float().to(device)
#         next_states_batch = torch.from_numpy(self.next_states[batch_idxs]).float().to(device)
#         advantages_batch = torch.from_numpy(self.advantages[batch_idxs]).float().to(device)
#         logprobs_batch = torch.from_numpy(self.logprobs[batch_idxs]).float().to(device)
        
#         return states_batch, actions_batch, rewards_batch, next_states_batch, advantages_batch, logprobs_batch
