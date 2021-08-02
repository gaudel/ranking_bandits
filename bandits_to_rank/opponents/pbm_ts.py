#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import random

import numpy as np
from numpy.random.mtrand import beta


from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index
from bandits_to_rank.tools.get_inference_model import GetSVD,GetMLE,GetOracle



class PBM_TS:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, nb_position, get_model_kappa, count_update=1, prior_s=0.5, prior_f=0.5):
        """

        :param nb_arms:
        :param nb_position:
        :param get_model_kappa: if get_model_kappa != Oracle, discount factors are inferred from logs every `lag` iterations
        :param lag:
        :param prior_s:

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>>discount_factor = [1, 0.9, 0.7]
        >>> player = PBM_TS_semi_oracle(nb_arms,nb_choices, discount_factor=discount_factor)

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
        >>> player = PBM_TS_semi_oracle(nb_arms,nb_choices, discount_factor=discount_factor)
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
        >>> player = PBM_TS_semi_oracle(nb_arms,nb_choices, discount_factor=discount_factor)
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
        self.nb_arms = nb_arms
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.nb_position = nb_position
        self.positions = np.arange(self.nb_position)
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
        self.nb_trials = 0

        # clean the log
        self.success = np.zeros([self.nb_arms, self.nb_position], dtype=np.int)
        self.place_view = np.zeros([self.nb_arms, self.nb_position], dtype=np.int)
        self.n_try = np.zeros(self.nb_arms, dtype=np.int) # number of times a proposal has been drawn for arm i's parameter
        self.n_drawn = np.zeros(self.nb_arms, dtype=np.int) # number of times arm i's parameter has been drawn

    def choose_next_arm(self):#### Attention resampling
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
        #print('pos', more_seen_position, 'for arm', i_arm)
        #print("prints:\n", self.place_view[i_arm, :])
        #print("clicks:\n", self.success[i_arm, :])
        #print("estimated iteration:", np.sum(self.place_view) / self.nb_position)
        #print("kappas:\n", self.discount_factor)
        #print("prints:\n", self.place_view)
        #print("clicks:\n", self.success)


        self.time_reject += n_inner_try
        return theta

    def update(self, propositions, rewards):
        # update model of kappa
        self.nb_trials += 1
        self.model_kappa.add_session(propositions, rewards)
        if self.nb_trials <= 100 or self.nb_trials % self.count_update == 0:
            self.model_kappa.learn()
            _, self.discount_factor = self.model_kappa.get_params()

        # update PBM_TS model
        self.place_view[propositions, self.positions] += 1
        self.success[propositions, self.positions] += rewards

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




def PBM_TS_semi_oracle(nb_arms, nb_position, discount_factor, prior_s=0.5, prior_f=0.5, count_update=1):
    """
    PBM_TS, where kappa is known.
    """
    return PBM_TS(nb_arms, nb_position, GetOracle(discount_factor), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def PBM_TS_Greedy_SVD(nb_arms, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    PBM_TS, where kappa is inferred assuming on rank-1 model (equivalent to PBM), with parameters inferred through SVD of empirical click-probabilities.
    """
    return PBM_TS(nb_arms, nb_position, GetSVD(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def PBM_TS_Greedy_MLE(nb_arms, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    PBM_TS, where kappa is inferred through MLE of empirical click-probabilities.
    """
    return PBM_TS(nb_arms, nb_position, GetMLE(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)





if __name__ == "__main__":
    import doctest

    doctest.testmod()
