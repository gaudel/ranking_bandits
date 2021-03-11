#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import random

import numpy as np
from numpy.random.mtrand import beta
import scipy.stats as st


from bandits_to_rank.sampling.pbm_inference import EM, SVD
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index



class PBM_TS:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, nb_positions=None, discount_factor=None, lag=1, prior_s=1, prior_f=1):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
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
            self.t = 0
            self.discount_factor = np.ones(self.nb_positions, dtype=np.float)

        # clean the log
        self.success = np.zeros([self.nb_arms, self.nb_positions], dtype=np.uint)
        self.place_view = np.zeros([self.nb_arms, self.nb_positions], dtype=np.uint)
        self.n_try = np.zeros(self.nb_arms, dtype=np.int) # number of times a proposal has been drawn for arm i's parameter
        self.n_drawn = np.zeros(self.nb_arms, dtype=np.int) # number of times arm i's parameter has been drawn

    def choose_next_arm(self):
        thetas = np.ones(self.nb_arms, dtype=np.float)
        self.time_reject = 0
        for i in range(self.nb_arms):
            thetas[i] = self._rejection_sampling(i)
        return order_theta_according_to_kappa_index(thetas, self.discount_factor), self.time_reject

    def _rejection_sampling(self, i_arm, max_try=1000):

        self.n_drawn[i_arm] += 1
        thetas = np.arange(0.00000001, 1, 0.01)

        # --- target distribution ---
        # - adaptive multiplicative constant -
        log_lik_target = np.zeros(len(thetas))
        for l, kappa in enumerate(self.discount_factor):
            log_lik_target += np.log(thetas) * (self.success[i_arm][l] + self.prior_s - 1)
            log_lik_target += np.log(1 - thetas * kappa) * (self.place_view[i_arm][l] - self.success[i_arm][l] + self.prior_f - 1)
        log_norm_target = np.log(np.sum(np.exp(log_lik_target-max(log_lik_target)))) + max(log_lik_target)

        # --- Reject sampling with PBM-TS proposal ---
        # - parameters -
        more_seen_position = np.argmax(self.place_view[i_arm])
        alpha = self.success[i_arm][more_seen_position] + self.prior_s
        #beta_param = self.place_view[i_arm][more_seen_position] + 2 - alpha ### Formule initiale
        beta_param = self.place_view[i_arm][more_seen_position] - self.success[i_arm][more_seen_position] + self.prior_f
        #print('pos', more_seen_position, 'for arm', i_arm)
        #print('alpha', alpha)
        #print('beta_param', beta_param)
        more_seen_kappa = self.discount_factor[more_seen_position]
        #print('more seen kappa', more_seen_kappa)
        #print('kappas', self.discount_factor)
        # proposal distribution

        # - adaptive multiplicative constant -
        log_lik_proposal = np.log(thetas*more_seen_kappa) * (alpha-1)
        log_lik_proposal += np.log(1 - thetas*more_seen_kappa) * (beta_param-1)
        log_norm_proposal = np.log(np.sum(np.exp(log_lik_proposal-max(log_lik_proposal)))) + max(log_lik_proposal)

        log_M = np.nanmax(log_lik_target-log_lik_proposal) - log_norm_target + log_norm_proposal

        # --- rejection sampling ---
        for _ in range(max_try):
            self.n_try[i_arm] += 1
            theta = beta(alpha, beta_param) / more_seen_kappa
            n_inner_try = 0
            while theta > 1 and n_inner_try < max_try:
                #print ('alpha',alpha)
                #print ('beta_param',beta_param)
                theta = beta(alpha, beta_param) / more_seen_kappa
                n_inner_try += 1
             
            if n_inner_try == max_try:
                print("Warning: unable to sample a value smaller than 1, with alpha =", alpha, ", beta =", beta_param, ", and kappa =", more_seen_kappa)
                theta = random()
            u = random()
            threshold = 0
            # target probability measure (up to a factor): P(theta | data)
            for l, kappa in enumerate(self.discount_factor):
                log_lik_target += np.log(theta) * (self.success[i_arm][l] + self.prior_s - 1)
                log_lik_target += np.log(1 - theta * kappa) * (
                        self.place_view[i_arm][l] - self.success[i_arm][l] + self.prior_f - 1)
            threshold -= log_M
            # proposal probability measure (up to a factor): P_beta(theta)
            threshold -= np.log(theta*more_seen_kappa) * (alpha-1)
            threshold -= np.log(1 - theta*more_seen_kappa) * (beta_param-1)
            if np.log(u) < threshold:
                self.time_reject += n_inner_try
                return theta
        #print("WARNING: PBM-TS rejection sampling stopped due to max_try, with alpha =", alpha, ", beta =", beta_param, ", kappa =", more_seen_kappa, ", and threshold =", np.exp(threshold))
        print('pos', more_seen_position, 'for arm', i_arm)
        #print("prints:\n", self.place_view[i_arm, :])
        #print("clicks:\n", self.success[i_arm, :])
        print("estimated iteration:", np.sum(self.place_view) / self.nb_positions)
        #print("kappas:\n", self.discount_factor)
        #print("prints:\n", self.place_view)
        #print("clicks:\n", self.success)


        self.time_reject += n_inner_try
        return theta

    def update(self, propositions, rewards):
        for pos in range(len(propositions)):
            item = propositions[pos]
            rew = rewards[pos]
            self.success[item][pos] += rew
            self.place_view[item][pos] += 1
        if not self.known_discount:
            self.learner.add_session(propositions, rewards)
            self.t += 1
            if self.t < 100 or self.t % self.lag == 0:
                self.learner.learn()
                self.discount_factor = self.learner.get_kappas()

    def get_theta_tild(self, k):
        S_k = sum(self.success[k])
        Ntild_k = sum(self.discount_factor * self.place_view[k])
        theta_tild = S_k / Ntild_k
        return theta_tild

    def get_param_estimation(self):
        thetas_estime = [self.get_theta_tild(k) for k in range(self.nb_arms)]
        return thetas_estime,self.discount_factor

    def reject_proportions(self):
        return (self.n_try - self.n_drawn)/self.n_try



if __name__ == "__main__":
    import doctest

    doctest.testmod()
