#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import log
from bandits_to_rank.tools.tools import swap, start_up, newton
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class sGRAB:
    """

    """

    def __init__(self, nb_arms, nb_positions, T, memory_size=np.inf):
        """
        Parameters
        ----------
        nb_arms
        nb_positions
        T
            number of iteration

        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.list_transpositions = [(0, 0)]
        for i in range(self.nb_positions):
            for j in range(i+1, self.nb_arms):
                self.list_transpositions.append((i, j))
        self.certitude = log(T)
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """

        # clean the log
        self.precision = 0
        self.running_t = 0
        self.extended_leader = [i for i in range(self.nb_arms)]; shuffle(self.extended_leader)

        self.kappa_thetas = np.zeros((self.nb_arms, self.nb_arms))
        self.times_kappa_theta = np.zeros((self.nb_arms, self.nb_positions))
        self.upper_bound_kappa_theta = np.ones((self.nb_arms, self.nb_arms))
        self.leader_count = defaultdict(self.empty)  # number of time each arm has been the leader

    @staticmethod
    def empty(): # to enable pickling
        return 0

    def choose_next_arm(self):
        (i, j) = (0, 0)
        if self.leader_count[tuple(self.extended_leader[:self.nb_positions])] % len(self.list_transpositions) > 0:
            delta_upper_bound_max = 0
            for (k, l) in self.list_transpositions:
                item_k, item_l = self.extended_leader[k], self.extended_leader[l]
                value = - self.upper_bound_kappa_theta[item_k, k] - self.upper_bound_kappa_theta[item_l, l] \
                        + self.upper_bound_kappa_theta[item_l, k] + self.upper_bound_kappa_theta[item_k, l]
                if (value > delta_upper_bound_max):
                    (i, j) = (k, l)
                    delta_upper_bound_max = value
        proposition = np.array(swap(self.extended_leader[:self.nb_positions], (i, j), self.extended_leader[self.nb_positions:]))
        return proposition, 0

    def update(self, propositions, rewards):
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

    def update_leader(self):
        """

        Returns
        -------

        Examples
        -------
        >>> import numpy as np
        >>> player = sGRAB(nb_arms=5, nb_positions=3, T=100)
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
