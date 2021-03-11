#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Proposals for Metropolis Hasting

Draw a potential next particle from previous one
"""

# Package

from __future__ import division
import math
import scipy.stats as st
import scipy as sp
from bandits_to_rank.data.Methode_Simulation_KappasThetasKnown import *
from random import uniform
from copy import deepcopy
from scipy.special import logit,expit



"""Proposal"""

class Proposal_RW:
    """
    Proposal_MA : initalisation of a MH algorithm with a radom walk

    :param sigma:
    :param theta_init:
    :return:
    """
    
    def __init__(self, sigma=0.3, k=0, on_theta=True):
        self.sigma = sigma
        self.on_theta = on_theta
        self.k = k
        self.reject_time_MA = 0
    
    def next_part(self, part_prev):
        
        part = deepcopy(part_prev)
        if self.on_theta: 
            part[0][self.k] = part[0][self.k] + np.random.normal(0, self.sigma)
        else:
            part[1][self.k] = part[1][self.k] + np.random.normal(0, self.sigma)
        return part
    
    def compute_rho(self, part_prev, part):
        return 1.
    
    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]
        >>> proposal = Proposal_RW(sigma=0.3, k=0, on_theta=True)

        >>> proposal.log_compute_rho(part1, part1)
        0.0

        >>> true_val = 0.
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = proposal.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True
        """
        return 0.0
    
    
class TruncatedGaussianRandomWalk():
    """
    Gaussian Random Walk truncated to the interval [a,b]
    """

    def __init__(self, a=0, b=1, sigma=0.3, k=0, on_theta=True):
        self.a = a
        self.b = b
        self.sigma = sigma
        if on_theta:
            self.i_param0 = 0
        else:
            self.i_param0 = 1
        self.i_param1 = k
        self.reject_time_MA = 0

    def next_part(self, part_prev):
        part = deepcopy(part_prev)
        prop = part[self.i_param0][self.i_param1] + np.random.normal(0, self.sigma)
        while prop < self.a or prop > self.b:
            prop = part[self.i_param0][self.i_param1] + np.random.normal(0, self.sigma)
            self.reject_time_MA +=1
        part[self.i_param0][self.i_param1] = prop
        return part

    def compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]

        >>> proposal = TruncatedGaussianRandomWalk(a=-10^9, b=10^9, sigma=0.3, k=0, on_theta=True)
        >>> true_val = 1.
        >>> val = proposal.compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True
        >>> true_val = 1.
        >>> val = proposal.compute_rho(part2, part1)
        >>> abs(val - true_val) < 1e-6
        True

        >>> proposal = TruncatedGaussianRandomWalk(a=0, b=1, sigma=0.3, k=0, on_theta=True)
        >>> true_val = 0.8222703
        >>> val = proposal.compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-5
        True
        >>> true_val = 1.216145
        >>> val = proposal.compute_rho(part2, part1)
        >>> abs(val - true_val) < 1e-6
        True
        """
        x_prev = part_prev[self.i_param0][self.i_param1]
        x = part[self.i_param0][self.i_param1]
        return (st.norm.cdf(self.b, loc=x_prev, scale=self.sigma) - st.norm.cdf(self.a, loc=x_prev, scale=self.sigma)) / (st.norm.cdf(self.b, loc=x, scale=self.sigma) - st.norm.cdf(self.a, loc=x, scale=self.sigma))

    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]

        >>> proposal = TruncatedGaussianRandomWalk(a=-10^9, b=10^9, sigma=0.3, k=0, on_theta=True)
        >>> true_val = 0.
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True
        >>> true_val = 0.
        >>> val = proposal.log_compute_rho(part2, part1)
        >>> abs(val - true_val) < 1e-6
        True

        >>> proposal = TruncatedGaussianRandomWalk(a=0, b=1, sigma=0.3, k=0, on_theta=True)
        >>> true_val = -0.1956862
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-5
        True
        >>> true_val = 0.1956862
        >>> val = proposal.log_compute_rho(part2, part1)
        >>> abs(val - true_val) < 1e-6
        True
        """
        x_prev = part_prev[self.i_param0][self.i_param1]
        x = part[self.i_param0][self.i_param1]
        return np.log(st.norm.cdf(self.b, loc=x_prev, scale=self.sigma) - st.norm.cdf(self.a, loc=x_prev, scale=self.sigma)) - np.log(st.norm.cdf(self.b, loc=x, scale=self.sigma) - st.norm.cdf(self.a, loc=x, scale=self.sigma))

    def logpdf(self, part_prev, part):
        x_prev = part_prev[self.i_param0][self.i_param1]
        x = part[self.i_param0][self.i_param1]
        return st.norm.logpdf(x, loc=x_prev, scale=self.sigma) - np.log(st.norm.cdf(self.b, loc=x_prev, scale=self.sigma) - st.norm.cdf(self.a, loc=x_prev, scale=self.sigma))


class Proposal_RW_logit:
    """
    Proposal_MA : initalisation of a MH algorithm with a radom walk
    For this particular setting the particule which shape is : [[thetas][kappas][thetas_tilt][kappas_tilt]] with theta = logit(theta_tild)

    :param sigma:
    :param theta_init:
    :return:
    """
    
    def __init__(self, sigma=0.3, k=0, on_theta=True):
        self.sigma = sigma
        self.on_theta = on_theta
        self.k = k
        self.reject_time_MA = 0
    
    def next_part(self, part_prev):   
        part = deepcopy(part_prev)
        if self.on_theta: 
            part[0][self.k] = part[0][self.k] + np.random.normal(0, self.sigma)
            part[0][self.k] = expit(part[0][self.k])
        else:
            part[1][self.k] = part[1][self.k] + np.random.normal(0, self.sigma)
            part[1][self.k] = expit(part[1][self.k])
        return part
   

    def compute_rho(self, part_prev, part):
        return 1.
    
    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]
        >>> proposal = Proposal_RW(sigma=0.3, k=0, on_theta=True)

        >>> proposal.log_compute_rho(part1, part1)
        0.0

        >>> true_val = 0.
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = proposal.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True
        """
        return 0.0
    
    
class Proposal_maxposition :
    """
    Proposal_XXXXXXX: initalisation of a MH algorithm with a radom walk

    :param sigma:
    :param theta_init:
    :return:
    """
    
    def __init__(self, dico, k=0, on_theta=True):
        self.on_theta = on_theta
        self.k = k
        self.reject_time_MA = 0
        
        if self.on_theta:
            self.po_max = int(dico['most_seen'][k])
            self.succ = dico['success'][k][self.po_max]
            self.fail =  dico['fail'][k][self.po_max]
        else : 
            self.po_max = int(dico['most_placed'][k])
            self.succ =  np.transpose(dico['success'])[k][self.po_max]
            self.fail =  np.transpose(dico['fail'])[k][self.po_max]
            
    
    def next_part(self, part_prev):
        part = deepcopy(part_prev)
        if self.on_theta: 
            theta_tild = st.beta(self.succ+1, self.fail+1).rvs()
            part[0][self.k] = theta_tild / part_prev[1][self.po_max]
        else:
            kappa_tild = st.beta(self.succ+1, self.fail+1).rvs()
            part[1][self.k] = kappa_tild / part_prev[0][self.po_max]
        return part
    
    def compute_rho(self, part_prev, part):
        if self.on_theta: 
            kappa_pos_max = part_prev[1][self.po_max]
            return st.beta(self.succ+1, self.fail+1).pdf(part_prev[0][self.k]*kappa_pos_max)/st.beta(self.succ+1, self.fail+1).pdf(part[0][self.k]*kappa_pos_max)
        else:
            theta_prod_max = part_prev[0][self.po_max]
            return st.beta(self.succ+1, self.fail+1).pdf(part_prev[1][self.k]*theta_prod_max)/st.beta(self.succ+1, self.fail+1).pdf(part[1][self.k]*theta_prod_max)
        
    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]
        >>> proposal = Proposal_maxposition({"success":[[8, 6, 2], [8, 8, 2]], "fail":[[2, 4, 8], [2, 4, 8]], "max_seen":[1,0]}, k=0, on_theta=True)

        >>> proposal.log_compute_rho(part1, part1)
        0.0

        >>> true_val = sp.stats.beta(6+1, 4+1).logpdf(0.8*0.8) - sp.stats.beta(6+1, 4+1).logpdf(0.5*0.8)
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = proposal.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True
        """
        if self.on_theta: 
            kappa_pos_max = part_prev[1][self.po_max]
            res = 0
            x_prev = part_prev[0][self.k] * kappa_pos_max
            res += sp.special.xlog1py(self.fail, -x_prev) + sp.special.xlogy(self.succ, x_prev)
            x_new = part[0][self.k] * kappa_pos_max
            res -= sp.special.xlog1py(self.fail, -x_new) + sp.special.xlogy(self.succ, x_new)
            return res
        else:
            theta_prod_max = part_prev[0][self.po_max]
            res = 0
            x_prev = part_prev[1][self.k] * theta_prod_max
            res += sp.special.xlog1py(self.fail, -x_prev) + sp.special.xlogy(self.succ, x_prev)
            x_new = part[1][self.k] * theta_prod_max
            res -= sp.special.xlog1py(self.fail, -x_new) + sp.special.xlogy(self.succ, x_new)
            return res
      
   
    def print_coef(self):
        print ('is theta', self.on_theta)
        print('S=',self.succ+1)
        print('F=',self.fail+1)

    def pdf_(self, part_prev):
        #print(part_prev)
        if self.on_theta: 
            kappa_pos_max = part_prev[1][self.po_max]
            #print(part_prev_theta)
            return st.beta(self.succ+1, self.fail+1).pdf(part_prev[0][self.k]*kappa_pos_max) * kappa_pos_max
        else:
            theta_prod_max = part_prev[0][self.po_max]
            return st.beta(self.succ+1, self.fail+1).pdf(part_prev[1][self.k]*theta_prod_max) * theta_prod_max
        
    def log_pdf_(self, part_prev):
        #print(part_prev)
        if self.on_theta: 
            kappa_pos_max = part_prev[1][self.po_max]
            #print(part_prev_theta)
            return st.beta(self.succ+1, self.fail+1).logpdf(part_prev[0][self.k]*kappa_pos_max) + math.log(kappa_pos_max)
        else:
            theta_prod_max = part_prev[0][self.po_max]
            return st.beta(self.succ+1, self.fail+1).logpdf(part_prev[1][self.k]*theta_prod_max) + math.log(theta_prod_max)
        
    def pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                kappa_pos_max = part_prev[1][self.po_max]
                #print(part_prev)
                #print(part_prev[0])
                x.append(part_prev[0][self.k])
                y.append(st.beta(self.succ+1, self.fail+1).pdf(part_prev[0][self.k]*kappa_pos_max) * kappa_pos_max)
        else:
            for part_prev in part_prev_list:
                theta_prod_max = part_prev[0][self.po_max]
                x.append(part_prev[1][self.k])
                y.append(st.beta(self.succ+1, self.fail+1).pdf(part_prev[1][self.k]*theta_prod_max) * theta_prod_max)
        return x,y
    
    def log_pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                kappa_pos_max = part_prev[1][self.po_max]
                #print(part_prev)
                #print(part_prev[0])
                x.append(part_prev[0][self.k])
                y.append(st.beta(self.succ+1, self.fail+1).logpdf(part_prev[0][self.k]*kappa_pos_max) + math.log(kappa_pos_max))
        else:
            for part_prev in part_prev_list:
                theta_prod_max = part_prev[0][self.po_max]
                x.append(part_prev[1][self.k])
                y.append(st.beta(self.succ+1, self.fail+1).logpdf(part_prev[1][self.k]*theta_prod_max) + math.log(theta_prod_max))
        return x,y
    
    

    
    
class Proposal_PseudoView :
 
    """ 
    Prior_maxposition : proposition, paper PBM_TS
    :param on_theta:
    :param k:
    :param reject_time_MA:
    :param kappa:
    :param succ:
    :param fail:
    :param all_success:
    :param pseudo_nb_draws:
    :param law:
    :return:
    """
    
    def __init__(self, dico, k=0, on_theta=True):
        self.on_theta = on_theta
        self.k = k
        self.reject_time_MA = 0
        
        if self.on_theta:
            kappa = dico['kappa']
            succ = dico['success'][k]
            fail =  dico['fail'][k]
            self.all_success = sum(succ)
            self.pseudo_nb_draws = sum((succ+fail)*kappa)
        else:
            theta = dico['theta']
            succ = np.transpose(dico['success'])[k]
            fail = np.transpose(dico['fail'])[k]
            self.all_success = sum(succ)
            self.pseudo_nb_draws = sum((succ + fail) * theta)
 
        self.law = st.beta(self.all_success + 1, max(self.pseudo_nb_draws-self.all_success, 0) + 1)
            
            
    def next_part(self, part_prev):
        part = deepcopy(part_prev)
        if self.on_theta: 
            part[0][self.k] = self.law.rvs()
        else:
            part[1][self.k] = self.law.rvs()
        return part
    
    def compute_rho(self, part_prev, part):
        if self.on_theta: 
            return self.law.pdf(part_prev[0][self.k])/self.law.pdf(part[0][self.k])
        else:
            return self.law.pdf(part_prev[1][self.k])/self.law.pdf(part[1][self.k])
    
    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> kappas = np.array([1, 0.8, 0.2])
        >>> part1 = [np.array([0.8, 0.5]), kappas]
        >>> part2 = [np.array([0.5, 0.5]), kappas]
        >>> proposal = Proposal_PseudoView({"success":np.array([[8, 6, 2], [8, 8, 2]]), "fail":np.array([[2, 4, 8], [2, 4, 8]]), "kappa":kappas}, k=0, on_theta=True)

        >>> proposal.log_compute_rho(part1, part1)
        0.0

        >>> true_val = sp.stats.beta(16+1, 20-16+1).logpdf(0.8) - sp.stats.beta(16+1, 20-16+1).logpdf(0.5)
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = proposal.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True
        """
        if self.on_theta:
            res = 0
            x_prev = part_prev[0][self.k]
            res += sp.special.xlog1py(max(self.pseudo_nb_draws-self.all_success, 0), -x_prev) + sp.special.xlogy(self.all_success, x_prev)
            x_new = part[0][self.k]
            res -= sp.special.xlog1py(max(self.pseudo_nb_draws-self.all_success, 0), -x_new) + sp.special.xlogy(self.all_success, x_new)
            return res
        else:
            res = 0
            x_prev = part_prev[1][self.k]
            res += sp.special.xlog1py(max(self.pseudo_nb_draws-self.all_success, 0), -x_prev) + sp.special.xlogy(self.all_success, x_prev)
            x_new = part[1][self.k]
            res -= sp.special.xlog1py(max(self.pseudo_nb_draws-self.all_success, 0), -x_new) + sp.special.xlogy(self.all_success, x_new)
            return res

    def print_coef(self):
        print('is theta', self.on_theta)
        print('S=', self.all_success+1)
        print('F=', max(self.pseudo_nb_draws-self.all_success, 0) + 1)


    def pdf_(self, part_prev):
        if self.on_theta: 
            return self.law.pdf(part_prev[0][self.k])
        else:
            return self.law.pdf(part_prev[1][self.k]) 
    
    def log_pdf_(self, part_prev):
        if self.on_theta: 
            return self.law.logpdf(part_prev[0][self.k])
        else:
            return self.law.logpdf(part_prev[1][self.k]) 
        
    def pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                x.append(part_prev[0][self.k])
                y.append(self.law.pdf(part_prev[0][self.k]))
        else:
            for part_prev in part_prev_list:
                x.append(part_prev[1][self.k])
                y.append(self.law.pdf(part_prev[1][self.k]))
        return x,y
    
    def log_pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                x.append(part_prev[0][self.k])
                y.append(self.law.logpdf(part_prev[0][self.k]))
        else:
            for part_prev in part_prev_list:
                x.append(part_prev[1][self.k])
                y.append(self.law.logpdf(part_prev[1][self.k]))
        return x,y


class Proposal_pseudoViewBis(Proposal_PseudoView):
    """
    draw a proposal for posterior value of theta[k] (respectively kappa[k]), assuming kappa (respc. theta) known.
    Follow the PeusidView idea of BC-MPTS

    While Proposal_vueProba requires kappa (resp. theta) to be given at instantiation, Proposal_vueProbaBis use the corresponding value in the previous particule.
    """

    def __init__(self, dico, k=0, on_theta=True):
        self.on_theta = on_theta
        self.k = k
        self.reject_time_MA = 0

        if self.on_theta:
            self.succ = dico['success'][k]
            self.fail = dico['fail'][k]
            self.all_success = sum(self.succ)
        else:
            self.succ = np.transpose(dico['success'])[k]
            self.fail = np.transpose(dico['fail'])[k]
            self.all_success = sum(self.succ)

    def next_part(self, part_prev):
        part = deepcopy(part_prev)
        if self.on_theta:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[1])
            part[0][self.k] = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
        else:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[0])
            part[1][self.k] = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
        return part

    def compute_rho(self, part_prev, part):
        if self.on_theta:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[1])
            law = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
            return law.pdf(part_prev[0][self.k]) / law.pdf(part[0][self.k])
        else:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[0])
            law = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
            return law.pdf(part_prev[1][self.k]) / law.pdf(part[1][self.k])

    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:

        >>> import numpy as np
        >>> import scipy as sp
        >>> kappas = np.array([1, 0.8, 0.2])
        >>> part1 = [np.array([0.8, 0.5]), kappas]
        >>> part2 = [np.array([0.5, 0.5]), kappas]
        >>> proposal = Proposal_pseudoViewBis({"success":np.array([[8, 6, 2], [8, 8, 2]]), "fail":np.array([[2, 4, 8], [2, 4, 8]])}, k=0, on_theta=True)

        >>> proposal.log_compute_rho(part1, part1)
        0.0

        >>> true_val = sp.stats.beta(16+1, 20-16+1).logpdf(0.8) - sp.stats.beta(16+1, 20-16+1).logpdf(0.5)
        >>> val = proposal.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = proposal.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True
        """
        if self.on_theta:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[1])
            res = 0
            x_prev = part_prev[0][self.k]
            res += sp.special.xlog1py(max(pseudo_nb_draws - self.all_success, 0), -x_prev) + sp.special.xlogy(
                self.all_success, x_prev)
            x_new = part[0][self.k]
            res -= sp.special.xlog1py(max(pseudo_nb_draws - self.all_success, 0), -x_new) + sp.special.xlogy(
                self.all_success, x_new)
            return res
        else:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[0])
            res = 0
            x_prev = part_prev[1][self.k]
            res += sp.special.xlog1py(max(pseudo_nb_draws - self.all_success, 0), -x_prev) + sp.special.xlogy(
                self.all_success, x_prev)
            x_new = part[1][self.k]
            res -= sp.special.xlog1py(max(pseudo_nb_draws - self.all_success, 0), -x_new) + sp.special.xlogy(
                self.all_success, x_new)
            return res

    def print_coef(self):
        print('is theta', self.on_theta)
        print('S=', self.succ)
        print('F=', self.fail)

    def pdf_(self, part_prev):
        if self.on_theta:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[1])
            law = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
            return law.pdf(part_prev[0][self.k])
        else:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[0])
            law = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
            return law.pdf(part_prev[1][self.k])

    def log_pdf_(self, part_prev):
        if self.on_theta:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[1])
            law = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
            return self.law.logpdf(part_prev[0][self.k])
        else:
            pseudo_nb_draws = sum((self.succ + self.fail) * part_prev[0])
            law = st.beta(self.all_success + 1, max(pseudo_nb_draws - self.all_success, 0) + 1).rvs()
            return self.law.logpdf(part_prev[1][self.k])

    def pdf_multiparticule(self, part_prev_list):
        x = []
        y = []
        if self.on_theta:
            for part_prev in part_prev_list:
                x.append(part_prev[0][self.k])
                y.append(self.pdf_(part_prev))
        else:
            for part_prev in part_prev_list:
                x.append(part_prev[1][self.k])
                y.append(self.pdf_(part_prev))
        return x, y

    def log_pdf_multiparticule(self, part_prev_list):
        x = []
        y = []
        if self.on_theta:
            for part_prev in part_prev_list:
                x.append(part_prev[0][self.k])
                y.append(self.log_pdf_(part_prev))
        else:
            for part_prev in part_prev_list:
                x.append(part_prev[1][self.k])
                y.append(self.log_pdf_(part_prev))
        return x, y


class Proposal():
    """Interface for Proposals"""

    def __init__(self):
        """Create a proposal"""
        raise NotImplementedError()










if __name__ == "__main__":
    import doctest

    doctest.testmod()
