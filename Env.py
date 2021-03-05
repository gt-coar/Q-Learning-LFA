#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numerical Experiment Environment
Same as Baird's counterexample in Errata of
Baird, Leemon. "Residual Algorithms:
Reinforcement Learning with Function Approximation."
22 Nov 95 version
"""
import numpy as np


class BairdEnv(object):

    def __init__(self, discount_factor, reward_function):
        # 0: state 1, 1: state 2, ..., 6: state 7
        self.state_space = [0, 1, 2, 3, 4, 5, 6]

        # 0: dashed action, 1: solid action
        self.action_space = [0, 1]

        # The dashed action results in state 1--6,
        # each with equal probability.
        # P_0 = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0],
        #        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0],
        #        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0],
        #        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0],
        #        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0],
        #        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0],
        #        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0]]

        # The solid action results in the state 7 w.p. 1
        # P_1 = [[0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 1]]

        self.P = {0: np.array(7 * [6 * [1. / 6.] + [0]]),
                  1: np.array(7 * [6 * [0] + [1]])}

        self.gamma = discount_factor
        self.current_state = None
        self.reward_function = reward_function

    def reset(self):
        self.current_state = np.random.randint(0, high=len(self.state_space))

    def step(self, action):
        reward = self.reward_function[self.current_state, action]
        next_state = np.random.choice(self.state_space,
                                      p=self.P[action][self.current_state, :])
        self.current_state = next_state

        return reward, next_state
