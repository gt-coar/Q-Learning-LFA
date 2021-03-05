#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute delta(pi) given
(1) underlying MDP
(2) begavior policy pi
(3) feature matrix
"""

import numpy as np
from numpy import linalg as LA
from itertools import product
from numpy.linalg import matrix_rank


def MC_stat_dist(P, pi, state_space, action_space):
    '''
    Find the stationary distribution for an MDP with a fixed policy,
    i.e., given transition probabilities P(s'|s, a) for all s,a and s'
    and begavior policy pi(a|s) for all s and a
    '''

    MC_TM = np.zeros([len(state_space), len(state_space)])

    # P_MC[s'|s] = sum_over_a(pi(a|s)*P[a][s'|s])
    for s in state_space:
        for s_prime in state_space:
            MC_TM[s, s_prime] = sum([pi[s, a] * P[a][s, s_prime]
                                     for a in action_space])

    print('Markov Chain Transition Matrix is')
    print(MC_TM)
    print()
    # find eigenvalues and eigvectors of P_MC
    eigvalues, eigvectors = LA.eig(MC_TM.T)

    # stationary distribution mu is
    # the eigenvector correspionding to eigenvalue==1
    mu = False
    for index, eigenvalue in enumerate(eigvalues):
        if abs(eigenvalue - 1.0) <= 1e-4:
            mu = eigvectors[:, index]
            # normalize mu such that mu is a distribution
            mu = mu / sum(mu)
            assert np.min(mu) > 0.0, "Not all components are positive"
            print('stationary distribution is')
            print(mu)
            print()
            return MC_TM, mu

    assert mu, "stationary distribution does not exist!"


def set_B(state_space, action_space):
    # set B = the set of stationary deterministic policies
    # equivalently, set B = (state space S)^{size of action space}
    return list(product(action_space, repeat=len(state_space)))


def state_action_to_indicies(state_space, action_space):
    # ordering of the state-action pair (s,a)
    # index starts from 0 to |S| * |A| - 1
    # outer loop: state space, inner loop: action space
    SA2Ind = {}
    ind = 0
    for s in state_space:
        for a in action_space:
            SA2Ind[(s, a)] = ind
            ind += 1

    return SA2Ind


def is_feature_matrix_fullrank(Phi):
    # check if the feature matrix Phi is full column rank
    if matrix_rank(Phi) == Phi.shape[1]:
        return True
    else:
        return False


def feature_matrix_constructor(SA2Ind):
    # construct the feature matrix by user input:
    # for each state-action pair (s,a),
    # the user needs to enter an array associated to it of size d
    # and use whitespace to seperate the array elements.
    Phi = []

    for (s, a) in SA2Ind:
        print("Enter the feature vector of the state-action pair ({}, {})".format(s, a))  # noqa line too long warning
        # take in a string of weights separated by a whitespace
        str_weights = input().split(' ')
        arr_weights = [float(weight) for weight in str_weights]
        Phi.append(arr_weights)

    return np.array(Phi)


def diagonal_matrix_D(pi, mu, SA2Ind):
    # diagonal matrix D with positive entries mu(s) * pi(s|a)
    diag_arr = []
    for (s, a) in SA2Ind:
        diag_arr.append(mu[s] * pi[s, a])

    return np.diag(diag_arr)


def matrix_Sigma_D(Phi, D):
    # construct matrix Sigma_D
    return Phi.T.dot(D).dot(Phi)


def diagonal_matrix_D_mu(mu, state_space):
    # construct diagonal matrix D_mu with entries mu(s)
    diag_arr = []
    for s in state_space:
        diag_arr.append(mu[s])

    return np.diag(diag_arr)


def matrix_Sigma_b(Phi, D_mu, b, SA2Ind):
    # construct matrix Sigma_b
    Phi_b = []
    for (s, a) in SA2Ind:
        if a == b[s]:
            Phi_b.append(Phi[SA2Ind[(s, a)], :].tolist())

    Phi_b = np.array(Phi_b)

    return Phi_b.T.dot(D_mu).dot(Phi_b)


def eigen_decomposition(Sigma_D):
    # eigenvalue-eigenvector decomposition sigma_D = Q Lambda Q^T
    # return diagonal matrix sqrt(Lambda)^{-1} and orthonormal matrix Q
    eigenvalues, Q = LA.eigh(Sigma_D)
    sqrt_Lambda_inv = np.diag(1.0 / np.sqrt(eigenvalues))

    return sqrt_Lambda_inv, Q


def final_matrix(Sigma_b, Sigma_D):
    # matrix M = sqrt(Lambda)^{-1} Q^T Sigma_b Q sqrt(Lambda)^{-1}
    sqrt_Lambda_inv, Q = eigen_decomposition(Sigma_D)

    return sqrt_Lambda_inv.dot(Q.T).dot(Sigma_b).dot(Q).dot(sqrt_Lambda_inv)


def largest_eigenvalue_reciprocal(M):
    # [lambda_max(M)]^{-1}
    return 1.0 / np.max(LA.eigvals(M))


def calculate_delta_pi(Sigma_D, B, Phi, D_mu, SA2Ind):
    lambda_min = np.infty

    for b in B:
        Sigma_b = matrix_Sigma_b(Phi, D_mu, b, SA2Ind)
        M = final_matrix(Sigma_b, Sigma_D)
        lambda_min = min(lambda_min, largest_eigenvalue_reciprocal(M))

    return lambda_min


if __name__ == "__main__":
    # test the module above

    state_space = [0, 1, 2, 3, 4, 5, 6]
    action_space = [0, 1]
    # P_a(s'|s)
    P = {0: np.array(7 * [6 * [1. / 6.] + [0]]),
         1: np.array(7 * [6 * [0] + [1]])}

    # behavior policy pi(a|s)
    pi = np.array(7 * [[0.5, 0.5]])

    # feature matrix Phi
    Phi = np.zeros([14, 14])
    Phi[0, 7] = Phi[2, 8] = Phi[4, 9] = Phi[6, 10] = Phi[8, 11] = Phi[10, 12] = Phi[12, 13] = 1.0  # noqa line too long warning
    Phi[1, 0] = Phi[3, 0] = Phi[5, 0] = Phi[7, 0] = Phi[9, 0] = Phi[11, 0] = Phi[13, 7] = 1.0  # noqa line too long warning
    Phi[1, 1] = Phi[3, 2] = Phi[5, 3] = Phi[7, 4] = Phi[9, 5] = Phi[11, 6] = Phi[13, 0] = 2.0  # noqa line too long warning

    MC_TM, mu = MC_stat_dist(P, pi, state_space, action_space)
    B = set_B(state_space, action_space)
    print('The set B has {} elements'.format(len(B)))
    print('or equivalently, there are {} stationary deterministic policies'.format(len(B)))  # noqa line too long warning
    print()
    SA2Ind = state_action_to_indicies(state_space, action_space)
    print("The feature matrix you entered is")
    print(Phi)
    print()
    assert is_feature_matrix_fullrank(Phi), "The feature matrix does not satisfy the independent condtion!"  # noqa line too long warning
    print('The feature matrix is full column rank')
    print()
    D = diagonal_matrix_D(pi, mu, SA2Ind)
    Sigma_D = matrix_Sigma_D(Phi, D)
    D_mu = diagonal_matrix_D_mu(mu, state_space)
    M_pi = calculate_delta_pi(Sigma_D, B, Phi, D_mu, SA2Ind)
    print('The minimal value delta(pi) of the MaxMin probelm is {}'.format(M_pi))
    print()
    print('By our theorem, if the discount factor gamma is less than {}'.format(np.sqrt(M_pi)))  # noqa line too long warning
    print('then we can guarantee that Q-learning with linear function approximation')  # noqa line too long warning
    print('under the above behavior policy converges with probability one')  # noqa line too long warning
