#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rate of convergence of Q-learning with linear
function approximation with diminishing step sizes.
"""

import numpy as np
from numpy import linalg as LA
from copy import deepcopy
import sys
sys.path.append('../')

from Env import BairdEnv
from Q_value_iteration import Q_ValueIteration
from delta_pi import state_action_to_indicies
from Q_learner_with_linear_func_approx import Linear_Q_learner

pi = np.array(7 * [[0.5, 0.5]])  # policy pi(a|s)
Phi = np.zeros([14, 14])  # feature matrix
Phi[0, 7] = Phi[2, 8] = Phi[4, 9] = Phi[6, 10] = Phi[8, 11] = Phi[10, 12] = Phi[12, 13] = 1.0
Phi[1, 0] = Phi[3, 0] = Phi[5, 0] = Phi[7, 0] = Phi[9, 0] = Phi[11, 0] = Phi[13, 7] = 1.0
Phi[1, 1] = Phi[3, 2] = Phi[5, 3] = Phi[7, 4] = Phi[9, 5] = Phi[11, 6] = Phi[13, 0] = 2.0

T = 1000000  # number of iterations
gamma = 0.7  # discount factor
np.random.seed(0)
reward_function = np.random.uniform(0, 1, (7, 2))  # reward function
initial_theta = np.random.uniform(low=-5., high=5., size=14).astype(np.float)  # initial weight vector

Env = BairdEnv(discount_factor=gamma, reward_function=reward_function)
SA2Ind = state_action_to_indicies(Env.state_space, Env.action_space)
Q_VI = Q_ValueIteration(Env)
Optimal_Weights = Q_VI.get_optimal_weights(Phi)

epsilon = 0.1
eta1 = 3000
xi1 = 1.
K = int(eta1 / epsilon)

xi2 = 0.8
eta2 = 0.1 * (eta1 / epsilon)**xi2

xi3 = 0.6
eta3 = 0.1 * (eta1 / epsilon)**xi3

xi4 = 0.4
eta4 = 0.1 * (eta1 / epsilon)**xi4

Num_Exp = 2
Exp_Settings = [(eta1, xi1), (eta2, xi2), (eta3, xi3), (eta4, xi4)]

for (eta, xi) in Exp_Settings:
    SE = np.zeros(T)

    for exp in range(Num_Exp):
        Q_learner = Linear_Q_learner(Env.state_space, Env.action_space, pi,
                                     Env.gamma, deepcopy(initial_theta),
                                     Phi, SA2Ind)
        Env.reset()
        print('xi = {}, Experiment No. {}'.format(xi, exp+1))
        for idx, t in enumerate(range(K, K + T)):
            s = Env.current_state
            a = Q_learner.sample_action(s)
            r, s_prime = Env.step(a)
            Q_learner.update_theta(s, a, s_prime, r, t, eta=eta, xi=xi)

            SE[idx] += LA.norm(Optimal_Weights - Q_learner.theta)**2

    MSE = SE / Num_Exp
    log_MSE = np.log(MSE)
    np.save('log_MSE_{}exps_xi{}.npy'.format(Num_Exp, xi), log_MSE)
