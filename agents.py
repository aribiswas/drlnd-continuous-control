#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains implementations of reinforcement learning algorithms.

Created: Aug 04, 2020
Rev: _

@author: abiswas
"""

import numpy
import torch
import torch.optim as optim
from utils import advantage_function, clipped_ppo_loss

class PPOAgent:
    """
    Implementation of actor-critic PPO agent with clipped loss function.
    """

    def __init__(self, actor, critic,\
        horizon=256,\
        discount_factor=0.99,\
        gae_factor=0.95,\
        learn_rate=0.01,\
        clip_factor=0.2):
        """
        Initialize a REINFORCE agent.

        Parameters
        ----------
        policy : torch.nn.Module
            A neural network representing the policy.
        trajectories : int, optional
            Number of trajectories or rollouts for collecting experiences. The default is 3.
        horizon : int, optional
            Number of steps per trajectory. The default is 256.
        gamma : float, optional
            Discount factor. The default is 0.99.
        alpha : float, optional
            Learn rate for the policy. The default is 0.01.

        Returns
        -------
        None.

        """
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.clip_factor = clip_factor
        self.learn_rate = learn_rate

        # set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # create two policy representations
        self.policy = actor.to(self.device)
        self.old_policy = actor.to(self.device)

        # create critic representation
        self.critic = critic.to(self.device)

        # initialize lists for storing experiences
        self.reset_buffers()

        # create an optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learn_rate)

        # initialize logs
        self.loss_log = [0]

        # initialize counters
        self.step_count = 0
        self.time_count = 0


    def reset_buffers(self):
        """
        Reset the experience buffers to empty.

        Parameters
        -------
        None.

        Returns
        -------
        None.
        """

        # initialize buffers
        self.states = numpy.empty([1, self.horizon])
        self.actions = numpy.empty([1, self.horizon])
        self.rewards = numpy.empty([1, self.horizon])
        self.dones = numpy.empty([1, self.horizon])
        self.probs = numpy.empty([1, self.horizon])


    def step(self, state, action, reward, done, prob):
        """
        Step the agent and update policy when sufficient trajectories are collected.

        Parameters
        ----------
        state : numpy array
            Current state.
        action : numpy array
            Action taken at the current state.
        reward : numpy array
            Reward for the action.
        done : numpy array
            Flag for episode termination.
        prob : numpy array
            Probability of action.

        Returns
        -------
        None.

        """
        # add experience to buffer
        self.states[self.time_count+1] = state
        self.actions[self.time_count+1] = action
        self.rewards[self.time_count+1] = reward
        self.dones[self.time_count+1] = done
        self.probs[self.time_count+1] = prob

        # increment step counter
        self.step_count += 1
        self.time_count += 1

        # if horizon is reached, learn from experiences
        if done or (self.step_count >= self.horizon):
            self.learn()
            self.reset_buffers()
            self.time_count = 0
        else:
            # keep logging the last loss
            self.loss_log.append(self.loss_log[-1])


    def learn(self):
        """
        Learn from collected experiences and update the policy.

        Returns
        -------
        None.

        """

        # convert everything to torch.Tensor
        states = torch.tensor(self.states, dtype=torch.float, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float, device=self.device)

        # compute V(s) and V(s+1)
        state_values = self.critic.get_value(states)
        next_state_values = torch.cat(state_values[1:], torch.from_numpy(np.zeros(1)).float().to(self.device))

        # get the old and new probabilities
        new_probs = torch.tensor(self.probs, dtype=torch.float, device=self.device)
        _, old_probs = self.old_policy.get_action(states)

        # compute the advantage
        advantage = advantage_function(states, rewards, state_values, next_state_values,\
            self.horizon, self.discount_factor, self.gae_factor)

        # compute the loss
        L_clip = clipped_ppo_loss(old_probs, new_probs, advantage, self.clip_factor)

        # do backward pass and update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log the loss
        self.loss_log.append(loss.detach().cpu().numpy())
