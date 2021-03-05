#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear_Q_learner: Q-learning with linear function approximation algorithm
"""


import numpy as np


class Linear_Q_learner(object):

    def __init__(self, state_space, action_space, policy,
                 discount_factor, initial_guess_theta,
                 feature_matrix, state_action2indices):

        self.S = state_space
        self.n = len(self.S)
        self.A = action_space
        self.m = len(self.A)
        self.gamma = discount_factor
        self.theta = initial_guess_theta
        self.Phi = feature_matrix
        self.SA2Ind = state_action2indices
        self.pi = policy

    def update_theta(self, s, a, s_prime, r, k,
                     constant_step_size=None,
                     eta=None, xi=None):

        if constant_step_size is None:
            step_size = eta / k**xi

        else:
            step_size = constant_step_size

        TD_error = r + self.gamma * \
            np.max([np.dot(self.Phi[self.SA2Ind[(s_prime, a_prime)]],
                           self.theta) for a_prime in self.A]) - \
            np.dot(self.Phi[self.SA2Ind[(s, a)]], self.theta)

        self.theta += step_size * TD_error * self.Phi[self.SA2Ind[(s, a)]]

    def sample_action(self, current_state):
        return np.random.choice(self.A, p=self.pi[current_state, :])
