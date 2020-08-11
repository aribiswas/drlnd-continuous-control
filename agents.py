#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains implementations of reinforcement learning algorithms.

Created: Aug 04, 2020
Rev: _

@author: abiswas
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import PPOMemory, random_batch_indices, advantage_function, compute_td_targets, clipped_ppo_loss

class PPOAgent:
    """
    Implementation of actor-critic PPO agent with clipped loss function.
    """

    def __init__(self, 
                 actor, 
                 critic,
                 horizon=256,
                 batch_size=64,
                 epochs=3,
                 discount_factor=0.99,
                 gae_factor=0.95,
                 actor_LR=0.001,
                 critic_LR=0.001,
                 clip_factor=0.2,
                 entropy_factor=0.01):
        
        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.clip_factor = clip_factor
        self.actor_LR = actor_LR
        self.critic_LR = critic_LR
        self.entropy_factor = entropy_factor

        # set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # create two policy representations
        self.actor = actor.to(self.device)

        # create critic representation
        self.critic = critic.to(self.device)

        # initialize a buffer for storing experiences
        self.buffer = PPOMemory(state_dim=self.actor.num_obs, 
                                act_dim=self.actor.num_act, 
                                max_len=horizon)

        # create an optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_LR)

        # initialize logs
        self.actor_loss_log = [0]
        self.critic_loss_log = [0]

        # initialize step counter
        self.step_count = 0


    def step(self, state, action, reward, next_state, done):
        
        # get the log probability pi(a|s)
        _, logp, _ = self.actor.pi(state, action)
        
        # add data to buffer
        self.buffer.add(state, action, reward, next_state, done, logp.detach().numpy())

        # increment counter
        self.step_count += 1

        # if horizon is reached, learn from experiences
        if self.horizon % self.step_count == 0:
            
            # first calculate advantages and update memory
            self.update_advantages()
            
            # train the actor and critic
            self.learn()
            
            # empty the memory
            self.buffer.reset()


    def learn(self):
        
        # train
        for epochs in range(self.epochs):
            
            # DEBUG
            #print('Epoch={:d}\n'.format(epochs))
            
            # create batch indices for training
            # The idea is to divide the experience data into chunks of size 
            # equal to batch_size
            batch_indices = random_batch_indices(self.batch_size, self.horizon)
            
            batch_ct = 0
            
            for batch_idx in batch_indices:
                
                # DEBUG
                #batch_ct += 1
                #print('Batch={:d}\n'.format(batch_ct))
            
                # sample batch from memory
                states, actions, rewards, next_states, advantages, old_logps = self.buffer.sample(batch_idx, self.device)
                
                # ---- TRAIN ACTOR ----
                
                # get the new policy
                pi, logps, entropy = self.actor.pi(states, actions)
               
                # compute the clipped loss
                L_clip = clipped_ppo_loss(old_logps, logps, advantages, self.clip_factor)
                
                # entropy bonus for exploration
                L_en = self.entropy_factor * entropy
                
                # total loss
                actor_loss = L_clip + L_en
                
                # update policy
                # TODO: clip gradients after backward pass, before optim step
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # ---- TRAIN CRITIC ----
                
                # compute td targets
                state_values = self.critic.get_value(states)
                next_state_values = self.critic.get_value(next_states)   # need grad here, so no detach
                targets = compute_td_targets(next_state_values, rewards, self.discount_factor, self.device)
                
                # critic loss
                critic_loss = F.mse_loss(state_values, targets)
    
                # do backward pass and update network
                # TODO: clip gradients after backward pass, before optim step
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
                # log the losses
                self.actor_loss_log.append(actor_loss.detach().cpu().numpy())
                self.critic_loss_log.append(critic_loss.detach().cpu().numpy())

        
    def update_advantages(self):
        
        states = self.buffer.states
        next_states = self.buffer.next_states
        rewards = self.buffer.rewards
        
        # compute V(s) and V(s+1)
        # advantages are not used in critic update, so we can detach
        state_values = self.critic.get_value(states).detach().numpy()
        next_state_values = self.critic.get_value(next_states).detach().numpy()
        
        # compute advantages
        adv = advantage_function(states, 
                                 rewards, 
                                 state_values, 
                                 next_state_values, 
                                 self.horizon, 
                                 self.discount_factor, 
                                 self.gae_factor)
                
        # TODO: normalize advantages
        
        # update memory
        self.buffer.advantages = adv
