#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle

import numpy as np
from math import sqrt,log
from itertools import product

from bandits_to_rank.sampling.pbm_inference import SVD
from bandits_to_rank.tools.tools import order_index_according_to_kappa



def order_index_according_to_kappa(indices, kappas):
    """

    :param indices:
    :param kappas:
    :return:

    >>> import numpy as np
    >>> index = np.array([5, 1, 3])
    >>> kappas = np.array([1, 0.9, 0.8])
    >>> order_index_according_to_kappa(index, kappas)
    array([5, 1, 3])

    >>> index = np.array([5, 1, 3])
    >>> kappas = np.array([1, 0.8, 0.9])
    >>> order_index_according_to_kappa(index, kappas)
    array([5, 3, 1])
    """

    nb_position = len(kappas)
    indice_kappa_ordonne = np.array(kappas).argsort()[::-1][:nb_position]
    res = np.ones(nb_position, dtype=np.int)
    nb_put_in_res = 0
    for i in indice_kappa_ordonne:
        res[i]=indices[nb_put_in_res]
        nb_put_in_res+=1
    return res



class TOP_RANK:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, T, horizon_time_known=True,doubling_trick_active=False, nb_positions=None, discount_factor=None, lag=1, prior_s=1, prior_f=1):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param discount_factor: if None, discount factors are inferred from logs every `lag` iterations
        :param lag:
        :param T: number of trials
        :param prior_s:
        :param prior_f:

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>> discount_factor = [1, 0.9, 0.7]
        >>> player = TOP_RANK(nb_arms, discount_factor=discount_factor)

        # function to assert choices have the right form
        >>> def assert_choices(choices, nb_choices):
        ...     assert len(choices) == nb_choices, "recommmendation list %r should be of size %d" % (str(choices), nb_choices)
        ...     assert len(np.unique(choices)) == nb_choices, "there is duplicates in recommmendation list %r" % (str(choices))
        ...     for pos in range(nb_choices):
        ...          assert 0 <= choices[pos] < nb_arms, "recommendation in position %d is out of bound in recommmendation list %r" % (pos, str(choices))

        # First choices should be random uniform
        >>> n_runs = 100
        >>> counts = np.zeros(nb_arms)
        >>> # try one choice
        >>> assert_choices(player.choose_next_arm(), nb_choices)
        >>> # try several choices
        >>> for _ in range(n_runs):
        ...     choices = player.choose_next_arm()
        ...     assert_choices(choices, nb_choices)
        ...     for arm in choices:
        ...         counts[arm] += 1
        >>> # almost uniform ?
        >>> assert np.all(np.abs(counts/nb_choices/n_runs - 1./nb_arms) < 0.1), str(counts/nb_choices/n_runs)


        # Other choices have to be coherent
        >>> n_runs = 100
        >>> nb_choices = 1
        >>> discount_factor = [1]
        >>> nb_arms = 2
        >>> player = TOP_RANK(nb_arms, discount_factor=discount_factor)
        >>> for _ in range(3):
        ...     player.update(np.array([0]), np.array([1]))
        ...     player.update(np.array([1]), np.array([0]))
        >>> # try one choice
        >>> assert_choices(player.choose_next_arm(), nb_choices)
        >>> # try several choices
        >>> counts = np.zeros(nb_arms)
        >>> for _ in range(n_runs):
        ...     choices = player.choose_next_arm()
        ...     assert_choices(choices, nb_choices)
        ...     for arm in choices:
        ...         counts[arm] += 1
        >>> # cover each arm
        >>> assert np.all(counts > 0), "%r" % str(counts)
        >>> # first arm is more drawn
        >>> assert np.all(counts[0] >= counts), "%r" % str(counts)
        >>> # second arm is less drawn
        >>> assert np.all(counts[1] <= counts), "%r" % str(counts)

        # Other choices have to be coherent
        >>> n_runs = 500
        >>> nb_choices = 1
        >>> discount_factor = [1]
        >>> nb_arms = 10
        >>> player = TOP_RANK(nb_arms, discount_factor=discount_factor)
        >>> for i in range(nb_arms):
        ...     for _ in range(5):
        ...         player.update(np.array([i]), np.array([1]))
        ...         player.update(np.array([i]), np.array([0]))
        >>> player.last_present = np.array([0])
        >>> for _ in range(5):
        ...     player.update(np.array([0]), np.array([1]))
        ...     player.update(np.array([1]), np.array([0]))
        >>> counts = np.zeros(nb_arms)
        >>> # try one choice
        >>> assert_choices(player.choose_next_arm(), nb_choices)
        >>> # try several choices
        >>> for _ in range(n_runs):
        ...     choices = player.choose_next_arm()
        ...     assert_choices(choices, nb_choices)
        ...     for arm in choices:
        ...         counts[arm] += 1
        >>> # cover each arm
        >>> assert np.all(counts > 0), "%r" % str(counts)
        >>> # first arm is more drawn
        >>> assert np.all(counts[0] >= counts), "%r" % str(counts)
        >>> # second arm is less drawn
        >>> assert np.all(counts[1] <= counts), "%r" % str(counts)
        """
        if (discount_factor is None) == (nb_positions is None):
            raise ValueError("One of both `discount_facor` and `nb_positions` has to be defined")
        if discount_factor is not None:
            self.known_discount = True
            self.discount_factor = discount_factor
            nb_positions = len(discount_factor)
        else:
            self.known_discount = False
            self.lag = lag
        
            
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.nb_positions = nb_positions
        self.nb_arms = nb_arms
        self.n_try = np.zeros(nb_arms, dtype=np.int) # number of times a proposal has been drawn for arm i's parameter
        self.n_drawn = np.zeros(nb_arms, dtype=np.int) # number of times arm i's parameter has been drawn

        self.c = 3.43 ## 4sqrt(2/pi)/erf(squrt(2))
        self.set_L = set([i for i in range(nb_arms)])

        if horizon_time_known:
            self.delta = self.delta_from_horizon(T)
            if doubling_trick_active:
                self.T_horizon = T
            else:
                self.T_horizon = -1
        else:
            self.delta = T
            self.T_horizon = -1
            if doubling_trick_active:
                raise ValueError("Doubling trick requires an initial time-horizon")

        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the model
        if not self.known_discount:
            self.learner = SVD(self.nb_arms, self.nb_positions)
            self.learner.nb_views = np.ones((self.nb_arms, self.nb_positions)) * (self.prior_s+self.prior_f)
            self.learner.nb_clicks = np.ones((self.nb_arms, self.nb_positions)) * self.prior_s
            self.discount_factor = np.ones(self.nb_positions, dtype=np.float)

        # clean the log
        self.time = 0
        self.clean_at_doubling_trick()

    def clean_at_doubling_trick(self):
        """ Clean log data at doubling trick """
        self.graph = set()
        self.partition = [self.set_L]
        self.s = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.n = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)

    def delta_from_horizon(self, T):
        return 1/float(T)

    def choose_next_arm(self):
        reco_full = []
        for part in self.partition:
            part_reco = list(part)
            shuffle(part_reco)
            reco_full = reco_full+part_reco
        reco_final = np.array(reco_full[:self.nb_positions])
        return order_index_according_to_kappa(reco_final, self.discount_factor), 0

    def get_reward_arm(self,i,propositions, rewards):
        propositions_list=list(propositions)
        if i in propositions_list:
            pos = propositions_list.index(i)
            rew = rewards[pos]
        else :
            rew = 0
        return rew

    def update_matrix_and_graph(self, propositions, rewards):
        # updates only for propositions belonging to the same part of the partition
        for Pc in self.partition:
            for i, j in product(Pc, Pc):
                # --- update matrix ---
                C_i = self.get_reward_arm(i, propositions, rewards)
                C_j = self.get_reward_arm(j, propositions, rewards)
                # print('rewards', C_i,C_j)
                self.s[i][j] += C_i - C_j
                self.n[i][j] += abs(C_i - C_j)

                # --- update graph ---
                if self.n[i][j]>0:
                    threshold = sqrt(2*self.n[i][j]*log((self.c/self.delta)*sqrt(self.n[i][j])))
                    if self.s[i][j] >= threshold:
                         self.graph.add((j, i))

    def build_P_td(self, set_done):
        P = set()
        substract = self.set_L ^ set_done
        for i in substract:
            is_weaker = False
            for j in substract:
                if (i, j) in self.graph:
                    is_weaker = True
            if not (is_weaker):
                P.add(i)
        return P

    def partition_arm(self):
        set_done = set()
        self.partition = []
        while self.set_L != set_done:
            P = self.build_P_td(set_done)
            self.partition.append(P)
            set_done = set_done.union(P)

    def update(self, propositions, rewards):
        self.time += 1
        if self.time == self.T_horizon:   #double-tricking
            self.T_horizon *= 2
            self.delta = self.delta_from_horizon(self.T_horizon/2)
            self.clean_at_doubling_trick()
            
        self.update_matrix_and_graph(propositions, rewards)
        self.partition_arm()
        if not self.known_discount:
            self.learner.add_session(propositions, rewards)
            if self.time < 100 or self.time % self.lag == 0:
                self.learner.learn()
                self.discount_factor = self.learner.get_kappas()
        self.time+=1

    def get_param_estimation(self):
        raise NotImplementedError()



if __name__ == "__main__":
    import doctest

    doctest.testmod()
