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
from model import StochasticActor, Critic
from model import DeterministicActor, QCritic
from utils import PPOMemory, discounted_rtg, gae, clipped_ppo_loss
from utils import GaussianNoise, OUNoise, ExperienceBuffer


# ========== DDPG Agent =============


class DDPGAgent:
    
    def __init__(self,
                 osize,
                 asize,
                 seed=0,
                 buffer_length=int(1e6),
                 batch_size=256,
                 gamma=0.99,
                 tau=0.01,
                 update_freq=3,
                 actor_LR=1e-3,
                 critic_LR=1e-3):
        """
        Initialize a Deep Deterministic Policy Gradient (DDPG) agent.

        Parameters
        ----------
        osize : number
            Number of observation elements.
        asize : number
            Number of action elements.
        seed : number, optional
            Random seed. The default is 0.
        buffer_length : number, optional
            Size of experience buffer. The default is int(1e6).
        batch_size : number, optional
            Batch size for training. The default is 256.
        gamma : number, optional
            Discount factor. The default is 0.99.
        tau : number, optional
            Target network smoothing factor. The default is 0.01.
        update_freq : number, optional
            Target network update frequency. The default is 3.
        actor_LR : number, optional
            Learn rate for actor network. The default is 1e-3.
        critic_LR : number, optional
            Learn rate for critic network. The default is 1e-3.

        """
        
        # parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.step_count = 0
        self.tau = tau
        self.actor_LR = actor_LR
        self.critic_LR = critic_LR
        self.update_freq = update_freq
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # initialize actor
        self.actor = DeterministicActor(osize, asize, seed).to(self.device)
        self.target_actor = DeterministicActor(osize, asize, seed)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # initialize critic
        self.critic = QCritic(osize, asize, seed).to(self.device)
        self.target_critic = QCritic(osize, asize, seed)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_LR)
        
        # Experience replay
        self.buffer = ExperienceBuffer(osize, asize, buffer_length)
        
        # Noise model
        #self.noise_model = GaussianNoise(asize, mean=0, std=0.5, stdmin=0.01, decay=5e-6)
        self.noise_model = OUNoise(size=[20,4], mean=0, mac=0.2, var=0.05, varmin=0.001, decay=5e-6, seed=0)
        
        # initialize logs
        self.actor_loss_log = [0]
        self.critic_loss_log = [0]
        self.noise_log = [0]
        
    
    def get_action(self, state, train=False):
        """
        Get the action by sampling from the policy. If train is set to True 
        then the action contains added noise.

        Parameters
        ----------
        state : numpy array or tensor
            State of the environment.
        train : boolean, optional
            Flag for train mode. The default is False.

        Returns
        -------
        action : numpy array
            Action with optional added noise.

        """
        
        with torch.no_grad():
            action = self.actor.mu(state).numpy()
            
        # If in train mode then add noise
        if train:
            noise = self.noise_model.step()
            action += noise
        
        # clip the action, just in case
        action = np.clip(action, -1, 1)
        
        return action
        
    
    def step(self, state, action, reward, next_state, done):
        """
        Step the agent, store experiences and learn.

        Parameters
        ----------
        state : numpy array
            State of the environment.
        action : numpy array
            Actions, given the states.
        reward : numpy array
            Reward obtained from the environment.
        next_state : numpy array
            Next states of the environment.
        done : numpy array
            Termination criteria.
            
        """
        
        # add experience to replay
        self.buffer.add(state, action, reward, next_state, done)
        
        # increase step count
        self.step_count += 1
        
        # learn from experiences
        if self.buffer.__len__() > self.batch_size:
            
            # create mini batch for learning
            experiences = self.buffer.sample(self.batch_size)
            
            # train the agent
            self.learn(experiences)
            
    
    
    def learn(self, experiences):
        """
        Train the actor and critic.

        Parameters
        ----------
        experiences : list
            Experiences (s,a,r,s1,d).


        """
        
        # unpack experience
        states, actions, rewards, next_states, dones = experiences
        
        # normalize rewards
        #rewards = (rewards - np.mean(self.buffer.rew_buf)) / (np.std(self.buffer.rew_buf) + 1e-5)
        
        # compute td targets
        with torch.no_grad():
            target_action = self.target_actor.mu(next_states)
            targetQ = self.target_critic.Q(next_states,target_action)
            y = rewards + self.gamma * targetQ * (1-dones)
        
        # compute local Q values
        Q = self.critic.Q(states, actions)
        
        # critic loss
        critic_loss = torch.mean((y-Q)**2)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # gradient clipping
        self.critic_optimizer.step()
        
        # freeze critic before policy loss computation
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # actor loss
        actor_loss = -self.critic.Q(states, self.actor.mu(states)).mean()
        
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # gradient clipping
        self.actor_optimizer.step()
        
        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True
            
        # log the loss and noise
        self.actor_loss_log.append(actor_loss.detach().cpu().numpy())
        self.critic_loss_log.append(critic_loss.detach().cpu().numpy())
        self.noise_log.append(np.mean(self.noise_model.x))
        
        # soft update target actor and critic
        if self.step_count % self.update_freq == 0:
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)
            
    
    def soft_update(self, target_model, model):
        """
        Soft update target networks.

        """
        with torch.no_grad():
            for target_params, params in zip(target_model.parameters(), model.parameters()):
                target_params.data.copy_(self.tau*params + (1-self.tau)*target_params.data)
    
    


# ========== PPO Agent =============

class PPOAgent:

    def __init__(self, 
                 osize, 
                 asize,
                 horizon=256,
                 batch_size=64,
                 discount_factor=0.99,
                 gae_factor=0.95,
                 actor_LR=0.001,
                 critic_LR=0.001,
                 clip_factor=0.2,
                 entropy_factor=0.01):
        """
        Initialize an actor-critic PPO agent with clipped loss function.
        
        """
        
        self.horizon = horizon
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.clip_factor = clip_factor
        self.actor_LR = actor_LR
        self.critic_LR = critic_LR
        self.entropy_factor = entropy_factor
        
        # set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
        # create actor and critic neural networks
        self.actor = StochasticActor(osize, asize, seed=0).to(self.device)
        self.critic = Critic(osize, seed=0).to(self.device)

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
        
        with torch.no_grad():
            
            # get the log probability pi(a|s)
            _, logp, _ = self.actor.pi(state, action)
            logp = logp.numpy()
            
            # compute V(s) and V(s+1)
            state_value = self.critic.get_value(state).numpy()
            next_state_value = self.critic.get_value(next_state).numpy()
        
        # add data to buffer
        self.buffer.add(state, action, reward, next_state, done, state_value, next_state_value, logp)

        # increment counters
        self.step_count += 1
        self.time_count += 1

        # if horizon is reached, learn from experiences
        if self.time_count >= self.horizon:
            
            # compute rewards-to-go and advantages post trajectory completion
            self.update_trajectory()
            
            # train actor and critic
            self.learn()
            
            # reset memory
            self.buffer.reset()
            self.time_count = 0


    def learn(self):
        
        pi_iters = 50  # number of actor adam steps
        v_iters = 50   # number of critic adam steps
        
        # train actor in multiple Adam steps
        for _ in range(pi_iters):
            
            # randomly sample batch of experiences
            batch_idxs = np.random.choice(self.batch_size, self.batch_size)
            states, actions, rewards, next_states, dones, advantages, _, _, discounted_rewards_to_go, old_logps = self.buffer.sample(batch_idxs)
            
            # get the new policy data, requires_grad is true
            pi, logps, entropy = self.actor.pi(states, actions)
            logps = logps.reshape((self.batch_size,1))  # reshape logps to batchsize x 1
           
            # compute the clipped loss
            L_clip, ratio = clipped_ppo_loss(old_logps, logps, advantages, self.clip_factor)
            
            # entropy bonus for exploration
            L_en = -self.entropy_factor * entropy
            
            # total loss
            actor_loss = L_clip + L_en
            
            # update policy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # gradient clipping
            self.actor_optimizer.step()
            
            # log the loss
            self.actor_loss_log.append(actor_loss.detach().cpu().numpy())
            self.ratio_log.append(ratio.detach().cpu().numpy())
            
            
        # train critic in multiple Adam steps
        for _ in range(v_iters):
            
            batch_idxs = np.random.choice(self.batch_size, self.batch_size)
            states, actions, rewards, next_states, dones, advantages, _, _, discounted_rewards_to_go, old_logps = self.buffer.sample(batch_idxs)
            
            
            # compute current state values V(s), requires_grad is true
            state_values = self.critic.get_value(states)
            
            # critic loss
            critic_loss = torch.mean((state_values - discounted_rewards_to_go) ** 2)
    
            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # gradient clipping
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
        #rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)
        
        # compute discounted rewards-to-go
        drtg = discounted_rtg(rewards, self.discount_factor)
        
        # compute advantages
        adv = gae(rewards, dones, state_values, next_state_values, self.discount_factor, self.gae_factor)
                
        # normalize advantages
        #adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-5)
        
        # update buffer
        self.buffer.rewards = rewards
        self.buffer.advantages = adv
        self.buffer.drtg = drtg
    
    
    
    
    
    
    
    
    
