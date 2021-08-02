#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
import numpy as np
from numpy import log, sqrt
from scipy.optimize import linear_sum_assignment

from bandits_to_rank.tools.tools import start_up, newton

class CombUCB1:
    """
    Yi Gai, Bhaskar Krishnamachari and Mingyan Liu
    On the Combinatorial Multi-Armed Bandit Problem with Markovian Rewards
    """

    def __init__(self, nb_arms, nb_positions, exploration_factor=1.):
        """
        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.positions_indices = np.arange(self.nb_positions)
        self.expl_factor = exploration_factor
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.mu_hats = np.zeros((self.nb_arms, self.nb_positions))
        self.n_try = np.zeros((self.nb_arms, self.nb_positions))
        self.nb_rounds = 1

    def choose_next_arm(self, epsilon=10**-5):
        row_ind, col_ind = linear_sum_assignment(- self.mu_hats
                                                 - sqrt(self.expl_factor*log(self.nb_rounds)/(self.n_try+epsilon))
                                                 )
        return row_ind[np.argsort(col_ind)], 0

    def update(self, propositions, rewards):
        # update statistics
        self.nb_rounds += 1
        self.n_try[propositions, self.positions_indices] += 1
        self.mu_hats[propositions, self.positions_indices] += (rewards - self.mu_hats[propositions, self.positions_indices]) / self.n_try[propositions, self.positions_indices]

class KL_CombUCB1:
    """
    Yi Gai, Bhaskar Krishnamachari and Mingyan Liu
    On the Combinatorial Multi-Armed Bandit Problem with Markovian Rewards

    Changes w.r.t original paper
    * finite time-horizon flavor
    * KL-like bound
    """

    def __init__(self, nb_arms, nb_positions, horizon):
        """
        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.positions_indices = np.arange(self.nb_positions)
        self.certitude = log(horizon)
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.mu_hats = np.zeros((self.nb_arms, self.nb_positions))
        self.n_try = np.zeros((self.nb_arms, self.nb_positions))
        self.upper_bound_mus = np.ones((self.nb_arms, self.nb_positions))

    def choose_next_arm(self):
        row_ind, col_ind = linear_sum_assignment(-self.upper_bound_mus)
        return row_ind[np.argsort(col_ind)], 0

    def update(self, propositions, rewards):
        # update statistics
        self.n_try[propositions, self.positions_indices] += 1
        self.mu_hats[propositions, self.positions_indices] += (rewards - self.mu_hats[propositions, self.positions_indices]) / self.n_try[propositions, self.positions_indices]

        # update upper-bounds (known horizon => update only on newly observed entries)
        for k, item_k in enumerate(propositions):
            kappa_theta, n = self.mu_hats[item_k, k], self.n_try[item_k, k]
            start = start_up(kappa_theta, self.certitude, n)
            self.upper_bound_mus[item_k, k] = newton(kappa_theta, self.certitude, n, start)

