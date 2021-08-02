#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""

from random import shuffle,random,choice

import numpy as np
from numpy.random.mtrand import beta
import scipy.stats as st
from scipy.optimize import fminbound, root_scalar
from math import sqrt,log

from bandits_to_rank.tools.tools import order_index_according_to_kappa
from bandits_to_rank.tools.get_inference_model import GetSVD, GetMLE, GetOracle



class PBM_PIE:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, epsilon, T, nb_position, get_model_kappa, count_update=1, prior_s=1, prior_f=1, is_shuffled =False):
        """

        :param nb_arms:
        :param epsilon:
        :param T:
        :param nb_position:
        :param discount_factor: if None, discount factors are inferred from logs every `count_update` iterations
        :param count_update:
        :param prior_s:

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>> discount_factor = [1, 0.9, 0.7]
        >>> player = PBM_PIE(nb_arms, discount_factor=discount_factor)

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
        if T < nb_arms:
             raise ValueError("Not enought Try")
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.positions = np.arange(self.nb_position)
        self.epsilon = epsilon

        self.time_reject = 0
        self.count_update = count_update
        self.get_model_kappa = get_model_kappa
        self.delta = (1+self.epsilon)*log(T)
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.rng = np.random.default_rng()
        self.is_shuffled = is_shuffled
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
        self.success = np.ones([self.nb_arms, self.nb_position], dtype=np.int)* self.prior_s
        self.place_view = np.ones([self.nb_arms, self.nb_position], dtype=np.int) * (self.prior_s + self.prior_f)
        self.n_try = np.zeros(self.nb_arms, dtype=np.int) # number of times a proposal has been drawn for arm i's parameter
        self.n_drawn = np.zeros(self.nb_arms, dtype=np.int) # number of times arm i's parameter has been drawn
        self.warm_up_list = [i for i in range(self.nb_arms)]
        if self.is_shuffled :
            shuffle(self.warm_up_list)
        self.is_warm_up = True
        self.top_L_index = []
    
            
    def kullback_leibler_divergence(self,p,q):
        #print('p:',p)
        #print('q:',q)
        return p*log(p/q)+(1-p)*log(abs((1-p)/(1-q)))
    
    def phi(self,q,k):
        sum_perf_q_as_theta = 0
        for l in range(self.nb_position):
            sum_perf_q_as_theta += self.place_view[k][l] * self.kullback_leibler_divergence(self.success[k][l] / self.place_view[k][l], q * self.discount_factor[l])
        return sum_perf_q_as_theta
    
  
    def build_potential_group(self):
        potential_group=[]
        for k in range(self.nb_position):
            if k not in self.top_L_index :
                theta_min = fminbound(self.phi, 0, 1,args =[k])
                if self.phi(theta_min,k)*self.phi(1,k) <0:
                    sol = root_scalar((lambda x: (self.phi(x,k)-self.delta)), bracket=[theta_min, 1], method='brentq')
                    U_k = sol.root ## fminbound((lambda x: (self.phi(x,k)-self.delta)), theta_min, 1)
                else : 
                    U_k = 0
                    
                if self.is_potential(U_k):
                    potential_group.append(k)
        return potential_group
    
    def get_theta_tild(self,k):
        S_k = sum(self.success[k])
        Ntild_k = sum(self.discount_factor * self.place_view[k])
        theta_tild = S_k/Ntild_k
        return theta_tild
    
    def is_potential(self,U_k):
        return U_k >= self.top_L_index[-1]
       
    def permute_circulaire(self,t,l):
        first = t
        l_result = l[first:]+l[0:first]
        return l_result
        
    def choose_next_arm(self):#### Attention resampling
        if self.nb_trials <=self.nb_arms*2 :
             #print(self.t-1%self.nb_arms)
            thetas_index = self.permute_circulaire((self.nb_trials -1)%self.nb_arms,self.warm_up_list)[:self.nb_position]
            #print(thetas_index)
            return order_index_according_to_kappa(thetas_index, self.discount_factor), self.time_reject
        
        thetas = [self.get_theta_tild(k) for k in range(self.nb_arms) ]
        self.top_L_index = np.array(thetas).argsort()[::-1][:self.nb_position]
        self.time_reject = 0
        
        B = self.build_potential_group()
        if len(B)!=0:
            if self.rng.random() < 1/2 :
                self.top_L_index[-1] = choice(B)
        
        return order_index_according_to_kappa(self.top_L_index, self.discount_factor), self.time_reject

    
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



def PBM_PIE_semi_oracle(nb_arms, epsilon, T,  nb_position, discount_factor, prior_s=0.5, prior_f=0.5, count_update=1):
    """
    PBM_PIE, where kappa is known.
    """
    return PBM_PIE(nb_arms, epsilon=epsilon, T=T,  nb_position=nb_position, get_model_kappa=GetOracle(discount_factor), prior_s=prior_s, prior_f=prior_f, count_update=count_update)


def PBM_PIE_Greedy_SVD(nb_arms, epsilon, T,  nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    PBM_PIE, where kappa is inferred assuming on rank-1 model (equivalent to PBM), with parameters inferred through SVD of empirical click-probabilities.
    """
    return PBM_PIE(nb_arms, epsilon=epsilon, T=T, nb_position=nb_position, get_model_kappa=GetSVD(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def PBM_PIE_Greedy_MLE(nb_arms, epsilon, T, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    PBM_PIE, where kappa is inferred through MLE of empirical click-probabilities.
    """
    return PBM_PIE(nb_arms, epsilon=epsilon, T=T, nb_position = nb_position, get_model_kappa=GetMLE(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
