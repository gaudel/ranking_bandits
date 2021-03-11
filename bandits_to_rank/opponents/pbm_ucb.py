#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import random

import numpy as np
from numpy.random.mtrand import beta
import scipy.stats as st
from math import sqrt,log

from bandits_to_rank.sampling.pbm_inference import EM, SVD
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index



class PBM_UCB:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, epsilon, nb_positions=None, discount_factor=None, lag=1, prior_s=1, prior_f=1):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param delta:
        :param nb_choice:
        :param discount_factor: if None, discount factors are inferred from logs every `lag` iterations
        :param lag:
        :param prior_s:

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>> discount_factor = [1, 0.9, 0.7]
        >>> player = PBM_TS(nb_arms, discount_factor=discount_factor)

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
        >>> player = PBM_TS(nb_arms, discount_factor=discount_factor)
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
        >>> player = PBM_TS(nb_arms, discount_factor=discount_factor)
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
        self.nb_arms = nb_arms
        self.epsilon = epsilon
        self.prior_s = prior_s
        self.prior_f = prior_f
        if discount_factor is not None:
            self.known_discount = True
            self.discount_factor = discount_factor
            self.nb_positions = len(discount_factor)
        else:
            self.known_discount = False
            self.lag = lag
            self.nb_positions = nb_positions

        self.time_reject = 0
        self.t = 1 
        self.clean()

    def clean(self):
        """ Clean log data.
        To be ran before playing a new game.
        """
        # clean the model
        if not self.known_discount:
            self.learner = SVD(self.nb_arms, self.nb_positions)
            self.learner.nb_views = np.ones((self.nb_arms, self.nb_positions)) * (self.prior_s+self.prior_f)
            self.learner.nb_clicks = np.ones((self.nb_arms, self.nb_positions)) * self.prior_s
            #self.t = 0
            self.discount_factor = np.ones(self.nb_positions, dtype=np.float)

        # clean the log
        self.success = np.zeros([self.nb_arms, self.nb_positions], dtype=np.int)
        self.place_view = np.zeros([self.nb_arms, self.nb_positions], dtype=np.int)
        self.n_try = np.zeros(self.nb_arms, dtype=np.int) # number of times a proposal has been drawn for arm i's parameter
        self.n_drawn = np.zeros(self.nb_arms, dtype=np.int) # number of times arm i's parameter has been drawn

    def get_theta_tild(self,k):
        S_k = sum(self.success[k])
        Ntild_k = sum(self.discount_factor * self.place_view[k])
        theta_tild = S_k/Ntild_k
        return theta_tild
    
    def get_bound(self,k):
        Ntild_k = sum(self.discount_factor * self.place_view[k])
        N_k = sum(self.place_view[k])
        delta = (1+self.epsilon)*log(self.t)
        bound = sqrt(N_k/Ntild_k)*sqrt(delta/(2*Ntild_k))
        return bound
        
    def choose_next_arm(self):
        thetas = np.ones(self.nb_arms, dtype=np.float)
        self.time_reject = 0
        for i in range(self.nb_arms):
            theta_tild = self.get_theta_tild(i)
            bound = self.get_bound(i)
            thetas[i] = theta_tild+bound
        return order_theta_according_to_kappa_index(thetas, self.discount_factor), self.time_reject

    
    def update(self, propositions, rewards):
        index_kappa_order = [i for i in range(self.nb_positions)] #np.array(self.discount_factor).argsort()[::-1][:self.nb_positions]
        self.t += 1
        self.place_view[propositions, index_kappa_order] += 1
        self.success[propositions, index_kappa_order] += rewards
        if not self.known_discount:
            self.learner.add_session(propositions, rewards)
            if self.t < 100 or self.t % self.lag == 0:
                self.learner.learn()
                self.discount_factor = self.learner.get_kappas()

    def get_param_estimation(self):
        thetas_estime = [self.get_theta_tild(k) for k in range(self.nb_arms)]
        return thetas_estime, self.discount_factor


    def reject_proportions(self):
        return (self.n_try - self.n_drawn)/self.n_try



if __name__ == "__main__":
    import doctest

    doctest.testmod()
