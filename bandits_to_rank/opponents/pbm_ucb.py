#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import random

import numpy as np
from numpy.random.mtrand import beta
import scipy.stats as st
from math import sqrt,log

from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index
from bandits_to_rank.tools.get_inference_model import GetSVD, GetMLE, GetOracle


class PBM_UCB:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, epsilon, nb_position, get_model_kappa, count_update=1, prior_s=1, prior_f=1):
        """
        One of both `discount_facor` and `nb_position` has to be defined.

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



        """
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.positions = np.arange(self.nb_position)
        self.epsilon = epsilon
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.count_update = count_update
        self.get_model_kappa = get_model_kappa
        self.time_reject = 0

        self.clean()

    def clean(self):
        """ Clean log data.
        To be ran before playing a new game.
        """
        # clean the model
        self.model_kappa = self.get_model_kappa()
        _, self.discount_factor = self.model_kappa.get_params()
        self.nb_trials = 1

        # clean the log
        self.success = np.zeros([self.nb_arms, self.nb_position], dtype=np.int)
        self.place_view = np.zeros([self.nb_arms, self.nb_position], dtype=np.int)
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
        delta = (1+self.epsilon)*log(self.nb_trials)
        bound = sqrt(N_k/Ntild_k)*sqrt(delta/(2*Ntild_k))
        return bound
        
    def choose_next_arm(self):#### Attention resampling
        thetas = np.ones(self.nb_arms, dtype=np.float)
        self.time_reject = 0
        for i in range(self.nb_arms):
            theta_tild = self.get_theta_tild(i)
            bound = self.get_bound(i)
            thetas[i] = theta_tild+bound
        return order_theta_according_to_kappa_index(thetas, self.discount_factor), self.time_reject

    
    def update(self, propositions, rewards):
        # update model of kappa
        self.nb_trials += 1
        self.model_kappa.add_session(propositions, rewards)
        if self.nb_trials <= 100 or self.nb_trials % self.count_update == 0:
            self.model_kappa.learn()
            _, self.discount_factor = self.model_kappa.get_params()

        # update PBM_UCB model
        self.place_view[propositions, self.positions] += 1
        self.success[propositions, self.positions] += rewards


    def get_param_estimation(self):
        thetas_estime = [self.get_theta_tild(k) for k in range(self.nb_arms)]
        return thetas_estime, self.discount_factor


    def reject_proportions(self):
        return (self.n_try - self.n_drawn)/self.n_try





def PBM_UCB_semi_oracle(nb_arms, epsilon, nb_position, discount_factor, prior_s=0.5, prior_f=0.5, count_update=1):
    """
    PBM_UCB, where kappa is known.
    """
    return PBM_UCB(nb_arms, epsilon, nb_position, GetOracle(discount_factor), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def PBM_UCB_Greedy_SVD(nb_arms, epsilon, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    PBM_UCB, where kappa is inferred assuming on rank-1 model (equivalent to PBM), with parameters inferred through SVD of empirical click-probabilities.
    """
    return PBM_UCB(nb_arms, epsilon,nb_position, GetSVD(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def PBM_UCB_Greedy_MLE(nb_arms, epsilon, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    PBM_UCB, where kappa is inferred through MLE of empirical click-probabilities.
    """
    return PBM_UCB(nb_arms, epsilon, nb_position, GetMLE(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)





if __name__ == "__main__":
    import doctest

    doctest.testmod()
