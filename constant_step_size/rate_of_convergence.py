#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rate of convergence of Q-learning with linear
function approximation with constant step size.
"""

import numpy as np
from numpy import linalg as LA
from copy import deepcopy
import sys
sys.path.append('../')

from Env import BairdEnv
from delta_pi import state_action_to_indicies
from Q_learner_with_linear_func_approx import Linear_Q_learner

reward_function = np.zeros((7, 2))  # reward function is identically zero
gamma = 0.7  # discount factor
Env = BairdEnv(discount_factor=gamma, reward_function=reward_function)
pi = np.array(7 * [[0.5, 0.5]])  # policy pi(a|s)
Phi = np.zeros([14, 14])  # feature matrix
Phi[0, 7] = Phi[2, 8] = Phi[4, 9] = Phi[6, 10] = Phi[8, 11] = Phi[10, 12] = Phi[12, 13] = 1.0
Phi[1, 0] = Phi[3, 0] = Phi[5, 0] = Phi[7, 0] = Phi[9, 0] = Phi[11, 0] = Phi[13, 7] = 1.0
Phi[1, 1] = Phi[3, 2] = Phi[5, 3] = Phi[7, 4] = Phi[9, 5] = Phi[11, 6] = Phi[13, 0] = 2.0
SA2Ind = state_action_to_indicies(Env.state_space, Env.action_space)

T = 30000
Repeat = 1000
epsilon = 0.01  # constant step-size
initial_theta = np.random.randint(-10, high=11, size=14).astype(np.float)  # initial weight vector

SE = []
for exp in range(Repeat):
    SE_exp = [LA.norm(initial_theta)**2.0]
    Q_learner = Linear_Q_learner(Env.state_space, Env.action_space, pi,
                                 Env.gamma, deepcopy(initial_theta),
                                 Phi, SA2Ind)
    Env.reset()  # reset initial state
    print('Experiment No. {}: the initial state of '.format(exp + 1) +
          'the environment is {}'.format(Env.current_state))
    for t in range(0, T):
        s = Env.current_state
        a = Q_learner.sample_action(s)
        r, s_prime = Env.step(a)
        Q_learner.update_theta(s, a, s_prime, r, t, constant_step_size=epsilon)
        SE_exp.append(LA.norm(Q_learner.theta)**2.0)

    SE.append(SE_exp)

SE = np.array(SE)
MSE = np.mean(SE, axis=0)
log_MSE = np.log(MSE)
np.save('log_MSE_{}exps_epsilon{}.npy'.format(Repeat, epsilon), log_MSE)
