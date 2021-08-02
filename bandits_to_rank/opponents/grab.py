#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import log
from bandits_to_rank.tools.tools import swap_full, start_up, newton
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class GRAB:
    """
    """

    def __init__(self, nb_arms, nb_positions, T, gamma, forced_initiation=False):
        """
        Parameters
        ----------
        nb_arms
        nb_positions
        T
            number of iteration
        gamma
            periodicity of force GRAB to play the leader
        forced_initiation
            constraint on the nb_arms first iteration to play
            a permutation in order to explore each item at each position at least once

        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.list_transpositions = [(0, 0)]
        self.gamma = gamma
        self.forced_initiation = forced_initiation

        self.certitude = log(T)
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """

        # clean the log
        self.precision = 0
        self.running_t = 0
        self.extended_leader = [i for i in range(self.nb_arms)]; shuffle(self.extended_leader)
        self.list_transpositions = [(0, 0)]

        self.kappa_thetas = np.zeros((self.nb_arms, self.nb_arms))
        self.times_kappa_theta = np.zeros((self.nb_arms, self.nb_positions))
        self.upper_bound_kappa_theta = np.ones((self.nb_arms, self.nb_arms))
        self.leader_count = defaultdict(self.empty)  # number of time each arm has been the leader

    @staticmethod
    def empty(): # to enable pickling
        return 0

    def choose_next_arm(self):
        (i, j) = (0, 0)
        if self.forced_initiation and (self.running_t < self.nb_arms):
            proposition = np.array([(self.running_t+i) % self.nb_arms for i in range(self.nb_positions)])
            return proposition, 0

        elif self.leader_count[tuple(self.extended_leader[:self.nb_positions])] % self.gamma > 0:
            delta_upper_bound_max = 0
            for (k, l) in self.list_transpositions:
                item_k, item_l = self.extended_leader[k], self.extended_leader[l]
                value = - self.upper_bound_kappa_theta[item_k, k] - self.upper_bound_kappa_theta[item_l, l] \
                        + self.upper_bound_kappa_theta[item_l, k] + self.upper_bound_kappa_theta[item_k, l]
                if (value > delta_upper_bound_max):
                    (i, j) = (k, l)
                    delta_upper_bound_max = value
        proposition = np.array(swap_full(self.extended_leader, (i, j), self.nb_positions))
        return proposition, 0

    def update(self, propositions, rewards):
        self.running_t += 1
        # update statistics
        self.leader_count[tuple(self.extended_leader[:self.nb_positions])] += 1
        for k in range(self.nb_positions):
            item_k = propositions[k]
            kappa_theta, n = self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k]
            kappa_theta, n = kappa_theta + (rewards[k] - kappa_theta) / (n + 1), n + 1
            start = start_up(kappa_theta, self.certitude, n)
            upper_bound = newton(kappa_theta, self.certitude, n, start)
            self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k] = kappa_theta, n
            self.upper_bound_kappa_theta[item_k, k] = upper_bound

        # update the leader L(n) (in the neighborhood of previous leader)
        self.update_leader()
        self.update_transition()

    def update_leader(self):
        """

        Returns
        -------

        Examples
        -------
        >>> import numpy as np
        >>> player = GRAB(nb_arms=5, nb_positions=3, T=100)
        >>> mu_hats = np.array([[0.625, 0.479, 0.268, 0., 0.],
        ...        [0.352, 0.279, 0.139, 0., 0.],
        ...        [0.585, 0.434, 0.216, 0., 0.],
        ...        [0.868, 0.655, 0.335, 0., 0.],
        ...        [0.292, 0.235, 0.108, 0., 0.]])
        >>> player.kappa_thetas = mu_hats
        >>> mu_hats[[2, 3, 1], np.arange(3)].sum()
        1.379
        >>> mu_hats[[3, 0, 2], np.arange(3)].sum()
        1.563
        >>> mu_hats[[3, 2, 0], np.arange(3)].sum()
        1.57

        >>> player.update_leader()
        >>> player.extended_leader
        array([3, 2, 0, 1, 4])

        """
        row_ind, col_ind = linear_sum_assignment(-self.kappa_thetas)
        self.extended_leader = row_ind[np.argsort(col_ind)]

    def update_transition(self):
        pi = np.argsort(-self.kappa_thetas[self.extended_leader[:self.nb_positions], np.arange(self.nb_positions)])
        self.list_transpositions = [(0, 0)]
        pi_extended = list(pi) + ([i for i in self.extended_leader if i not in pi])
        for i in range(self.nb_arms - 1):
            if i < self.nb_positions:
                self.list_transpositions.append((pi_extended[i], pi_extended[i + 1]))
            else:
                self.list_transpositions.append((pi_extended[self.nb_positions - 1], pi_extended[i + 1]))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
