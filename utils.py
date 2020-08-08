#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains utility functions for reinforcement learning algorithms.

Created: Aug 05, 2020
Revised: _

@author: abiswas
"""

def advantage_function(states, rewards, state_values, next_state_values, horizon, gamma, gaelambda):
    """
    Compute the GAE advantage estimates.

    Parameters
    -------
    states: torch.Tensor or numpy.array
        Current states (s).
    rewards: torch.Tensor or numpy.array
        Current rewards.
    state_values: torch.Tensor or numpy.array
        state values V(s).
    next_state_values: torch.Tensor or numpy.array
        state values V(s+1).
    horizon: number
        Time horizon for advantage estimate.
    gamma: number
        Discount factor.
    lambda: number
        GAE factor.

    Returns
    -------
    advantage: torch.Tensor or numpy.array
        Advantage estimates.
    """

    coeffs = [(gamma*gaeambda)**j for j in range(horizon)]
    td_errors = rewards + gamma*next_state_values - state_values
    advantage = [c*d for c,d in zip(coeffs,td_errors)]

    return advantage


def clipped_ppo_loss(old_probs, new_probs, advantage, clip_factor):
    """
    Compute clipped surrogate loss.

    Returns
    -------
    loss : torch.Tensor
        Policy loss.

    """

    # policy ratio
    ratio = new_probs / old_probs
    ratio = torch.clamp(ratio, 1-clip_factor, 1+clip_factor)

    # loss
    return min(ratio*advantage, g)
