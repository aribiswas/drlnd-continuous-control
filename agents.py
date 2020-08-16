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
from model import StochasticActor, Critic
from utils import PPOMemory, discounted_rtg, gae, clipped_ppo_loss

class PPOAgent:
    """
    Implementation of actor-critic PPO agent with clipped loss function.
    """

    def __init__(self, 
                 osize, 
                 asize,
                 horizon=256,
                 discount_factor=0.99,
                 gae_factor=0.95,
                 actor_LR=0.001,
                 critic_LR=0.001,
                 clip_factor=0.2,
                 entropy_factor=0.01):
        
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.clip_factor = clip_factor
        self.actor_LR = actor_LR
        self.critic_LR = critic_LR
        self.entropy_factor = entropy_factor
        
        # create actor and critic neural networks
        actor = StochasticActor(osize, asize, seed=0)
        critic = Critic(osize, seed=0)

        # set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # create actor and critic representations
        self.actor = actor.to(self.device)
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
        self.ratio_log = [0]

        # initialize step counter
        self.step_count = 0
        self.time_count = 0
        
        
    def get_action(self, state):
        """
        Sample action from the current policy.

        Parameters
        ----------
        state : numpy array or torch tensor
            State of the environment.

        Returns
        -------
        action : numpy array
            Actions sampled from the policy, bounded in [-1,1].

        """
        
        with torch.no_grad():
            
            # get the policy
            dist, _, _ = self.actor.pi(state)
    
            # sample action
            action = dist.sample()
            action = torch.clamp(action, -1, 1)  # limit actions to [-1,1]

        return action.numpy()


    def step(self, state, action, reward, next_state, done):
        
        # get the log probability pi(a|s)
        _, logp, _ = self.actor.pi(state, action)
        
        # compute V(s) and V(s+1)
        state_value = self.critic.get_value(state).detach().numpy()
        next_state_value = self.critic.get_value(next_state).detach().numpy()
        
        # add data to buffer
        self.buffer.add(state, action, reward, next_state, done, state_value, next_state_value, logp.detach().numpy())

        # increment counters
        self.step_count += 1
        self.time_count += 1

        # if horizon is reached, learn from experiences
        if self.time_count >= self.horizon:
            
            # compute rewards-to-go and advantages post trajectory
            self.update_trajectory()
            
            # train actor and critic
            self.learn()
            
            # reset memory
            self.buffer.reset()
            self.time_count = 0


    def learn(self):
        
        states, actions, rewards, next_states, dones, advantages, _, _, discounted_rewards_to_go, old_logps = self.buffer.get()
        
        pi_iters = 50
        v_iters = 50
        
        # train actor in multiple Adam steps
        for _ in range(pi_iters):
            
            # get the new policy data, requires_grad is true
            pi, logps, entropy = self.actor.pi(states, actions)
            logps = logps.reshape((self.horizon,1))  # reshape logps to batchsize x 1
           
            # compute the clipped loss
            L_clip, ratio = clipped_ppo_loss(old_logps, logps, advantages, self.clip_factor)
            
            # entropy bonus for exploration
            L_en = -self.entropy_factor * entropy
            
            # total loss
            actor_loss = L_clip + L_en
            
            # update policy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # gradient clipping
            self.actor_optimizer.step()
            
            # log the loss
            self.actor_loss_log.append(actor_loss.detach().cpu().numpy())
            self.ratio_log.append(ratio.detach().cpu().numpy())
            
            
        # train critic in multiple Adam steps
        for _ in range(v_iters):
            
            # compute current state values V(s), requires_grad is true
            state_values = self.critic.get_value(states)
            
            # critic loss
            critic_loss = torch.mean((state_values - discounted_rewards_to_go) ** 2)
    
            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # gradient clipping
            self.critic_optimizer.step()
        
            # log the loss
            self.critic_loss_log.append(critic_loss.detach().cpu().numpy())
            
            
        
    def update_trajectory(self):
        
        states = self.buffer.states
        next_states = self.buffer.next_states
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        state_values = self.buffer.state_values
        next_state_values = self.buffer.next_state_values
        
        # normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)
        
        # compute discounted rewards-to-go
        drtg = discounted_rtg(rewards, self.discount_factor)
        
        # compute advantages
        adv = gae(rewards, dones, state_values, next_state_values, self.discount_factor, self.gae_factor)
                
        # normalize advantages
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-5)
        
        # update buffer
        self.buffer.rewards = rewards
        self.buffer.advantages = adv
        self.buffer.drtg = drtg
