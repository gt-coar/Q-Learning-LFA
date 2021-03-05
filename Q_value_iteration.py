#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q value iteration algorithm
"""

import numpy as np
import itertools
from Env import BairdEnv


class Q_ValueIteration(object):

    def __init__(self, Env, threshold=1e-6):

        self.Env = Env
        self.threshold = threshold

        # Q value inilization
        self.Q = np.zeros([len(self.Env.state_space),
                           len(self.Env.action_space)])

    def run(self):
        state_action_pairs = list(itertools.product(self.Env.state_space,
                                                    self.Env.action_space))
        diff = np.ones([len(self.Env.state_space),
                        len(self.Env.action_space)]) * np.inf

        while np.max(diff) > self.threshold:
            for (s, a) in state_action_pairs:
                # New_Q(s,a) = R(s,a) + \
                # gamma * sum_s' P(s'|s,a) * max_a' Q(s', a')
                new_Q_s_a = self.Env.reward_function[s, a] + self.Env.gamma * \
                    np.sum([self.Env.P[a][s, s_prime] *
                            np.max(self.Q[s_prime, :])
                            for s_prime in self.Env.state_space])

                diff[s, a] = abs(new_Q_s_a - self.Q[s, a])
                self.Q[s, a] = new_Q_s_a

    def get_optimal_weights(self, Phi):
        self.run()
        return np.linalg.solve(Phi, self.Q.flatten())


if __name__ == "__main__":
    # test the module above
    discount_factor = 0.7
    reward_function = np.random.uniform(0, 1, (7, 2))
    Q_VI = Q_ValueIteration(BairdEnv(discount_factor, reward_function))
    Phi = np.zeros([14, 14])  # feature matrix
    Phi[0, 7] = Phi[2, 8] = Phi[4, 9] = Phi[6, 10] = Phi[8, 11] = Phi[10, 12] = Phi[12, 13] = 1.0  # noqa line too long warning
    Phi[1, 0] = Phi[3, 0] = Phi[5, 0] = Phi[7, 0] = Phi[9, 0] = Phi[11, 0] = Phi[13, 7] = 1.0  # noqa line too long warning
    Phi[1, 1] = Phi[3, 2] = Phi[5, 3] = Phi[7, 4] = Phi[9, 5] = Phi[11, 6] = Phi[13, 0] = 2.0  # noqa line too long warning
    print(Q_VI.get_optimal_weights(Phi))
    print(Phi)
    print(Q_VI.Q.flatten())
