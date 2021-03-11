#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Target """

# Packages
from __future__ import division
import math
import scipy.stats as st
import scipy as sp
from bandits_to_rank.data.Methode_Simulation_KappasThetasKnown import *

""" Target"""

class Target :
    """
    Target_XXXXXXX: initalisation of a MH algorithm with a radom walk

    :param sigma:
    :param theta_init:
    :return:
    """
    
    def __init__(self, dico, k=0,on_theta=True):
        self.on_theta = on_theta
        self.k = k
        self.dico = dico
        if self.on_theta:
            self.succ = dico['success'][k]
            self.fail =  dico['fail'][k]
        else : 
            self.succ =  np.transpose(dico['success'])[k]
            self.fail =  np.transpose(dico['fail'])[k]
            
    
    def compute_rho(self, part_prev, part):
        if self.on_theta:
            if part[0][self.k] > 1:
                return 0.
            if part[0][self.k] < 0:
                return 0.
            res = 1
            #print ('part[1]',part[1])
            for l in range(part[1].shape[0]):
                #print (part[1].shape[0])
                kappa = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(part[0][self.k]*kappa) / st.beta(succ_l+1, fail_l+1).pdf(part_prev[0][self.k]*kappa)
            return res
        else:
            if part[1][self.k] > 1:
                return 0.
            if part[1][self.k] < 0:
                return 0.
            res = 1
            for l in range(part[0].shape[0]):
                theta = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(part[1][self.k]*theta) / st.beta(succ_l+1, fail_l+1).pdf(part_prev[1][self.k]*theta)
            return res
        
        
    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:


        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part3 = [np.array([-0.1, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part4 = [np.array([1.1, 0.5]), np.array([1, 0.8, 0.2])]
        >>> target = Target({"success":[[8, 6, 2], [8, 8, 2]], "fail":[[2, 4, 8], [2, 4, 8]]}, k=0, on_theta=True)

        >>> target.log_compute_rho(part1, part1)
        0.0

        >>> true_val = sp.stats.beta(8+1, 2+1).logpdf(0.5*1) + sp.stats.beta(6+1, 4+1).logpdf(0.5*0.8) + sp.stats.beta(2+1, 8+1).logpdf(0.5*0.2) - sp.stats.beta(8+1, 2+1).logpdf(0.8*1) - sp.stats.beta(6+1, 4+1).logpdf(0.8*0.8) - sp.stats.beta(2+1, 8+1).logpdf(0.8*0.2)
        >>> val = target.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = target.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True

        >>> target.log_compute_rho(part1, part3)
        -inf

        >>> target.log_compute_rho(part1, part4)
        -inf
        """
        if self.on_theta:
            if part[0][self.k] > 1:
                return - np.inf
            if part[0][self.k] < 0:
                return - np.inf
            res = 0
            #print ('part[1]',part[1])
            for l in range(part[1].shape[0]):
                #print (part[1].shape[0])
                kappa = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                x_new = part[0][self.k]*kappa
                res += sp.special.xlog1py(fail_l, -x_new) + sp.special.xlogy(succ_l, x_new)
                x_prev = part_prev[0][self.k]*kappa
                res -= sp.special.xlog1py(fail_l, -x_prev) + sp.special.xlogy(succ_l, x_prev)
            return res
        else:
            if part[1][self.k] > 1:
                return - np.inf
            if part[1][self.k] < 0:
                return - np.inf
            res = 0
            for l in range(part[0].shape[0]):
                theta = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                x_new = part[1][self.k]*theta
                res += sp.special.xlog1py(fail_l, -x_new) + sp.special.xlogy(succ_l, x_new)
                x_prev = part_prev[1][self.k]*theta
                res -= sp.special.xlog1py(fail_l, -x_prev) + sp.special.xlogy(succ_l, x_prev)
            return res
    
    def pdf_(self, part_prev):
        res = 1
        if self.on_theta:
            if part_prev[0][self.k] > 1:
                return 0.
            if part_prev[0][self.k] < 0:
                return 0.
            for l in range(part_prev[1].shape[0]):
                kappa_mono = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(part_prev[0][self.k ]*kappa_mono) * kappa_mono
            return res
        else:
            if part_prev[1][self.k] > 1:
                return 0.
            if part_prev[1][self.k] < 0:
                return 0.
            for l in range(part_prev[0].shape[0]):
               
                theta_mono = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(theta_mono*part_prev[1][self.k]) * theta_mono
                #print (l, res)
            return res
        
    def log_pdf_(self, part_prev):
        res = 0
        if self.on_theta:
            if part_prev[0][self.k] > 1:
                return 0.
            if part_prev[0][self.k] < 0:
                return 0.
            for l in range(part_prev[1].shape[0]):
                kappa_mono = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res += st.beta(succ_l+1, fail_l+1).logpdf(part_prev[0][self.k ]*kappa_mono) + math.log(kappa_mono)
            return res
        else:
            if part_prev[1][self.k] > 1:
                return 0.
            if part_prev[1][self.k] < 0:
                return 0.
            for l in range(part_prev[0].shape[0]):
                theta_mono = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res += st.beta(succ_l+1, fail_l+1).logpdf(theta_mono*part_prev[1][self.k]) + math.log(theta_mono) 
                #print (l, res)
            return res

       
    def pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                if part_prev[0][self.k] > 1:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                if part_prev[0][self.k] < 0:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                else :
                    res = 1
                    for l in range(part_prev[1].shape[0]):
                        #print ('l',l)
                        #print ('part_prev[1]',part_prev[1])
                        kappa_mono = part_prev[1][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res *= st.beta(succ_l+1, fail_l+1).pdf(part_prev[0][self.k]*kappa_mono) * kappa_mono
                    #print ('res',res)
                    x.append(part_prev[0][self.k])
                    y.append(res)
                
        else:
            for part_prev in part_prev_list:
                if part_prev[1][self.k] > 1:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                if part_prev[1][self.k] < 0:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                else:
                    res = 1
                    for l in range(part_prev[0].shape[0]):
                        theta_mono = part_prev[0][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res *= st.beta(succ_l+1, fail_l+1).pdf(theta_mono*part_prev[1][self.k]) * theta_mono
                    x.append(part_prev[1][self.k])
                    y.append(res)
        #print ('x',x,'y',y)
        return x,y
    
    def log_pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                if part_prev[0][self.k] > 1:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                if part_prev[0][self.k] < 0:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                else :
                    res = 0
                    for l in range(part_prev[1].shape[0]):
                        #print ('l',l)
                        #print ('part_prev[1]',part_prev[1])
                        kappa_mono = part_prev[1][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res += st.beta(succ_l+1, fail_l+1).logpdf(part_prev[0][self.k]*kappa_mono) + math.log(kappa_mono)
                    #print ('res',res)
                    x.append(part_prev[0][self.k])
                    y.append(res)
                
        else:
            for part_prev in part_prev_list:
                if part_prev[1][self.k] > 1:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                if part_prev[1][self.k] < 0:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                else:
                    res = 0
                    for l in range(part_prev[0].shape[0]):
                        theta_mono = part_prev[0][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res += st.beta(succ_l+1, fail_l+1).logpdf(theta_mono*part_prev[1][self.k]) + math.log(theta_mono)
                    x.append(part_prev[1][self.k])
                    y.append(res)
        #print ('x',x,'y',y)
        return x,y
        
         
        

    
""" Target"""

class Target_TS :
    """
    Target_XXXXXXX: initalisation of a MH algorithm with a radom walk

    :param sigma:
    :param theta_init:
    :return:
    """
    
    def __init__(self, success, failure, k=0,on_theta=True):
        self.on_theta = on_theta
        self.k = k
        if self.on_theta:
            self.succ = success[k]
            self.fail = failure[k]
        else : 
            self.succ =  np.transpose(success)[k]
            self.fail =  np.transpose(failure)[k]
            
    
    def compute_rho(self, part_prev, part):
        if self.on_theta:
            if part[0][self.k] > 1:
                return 0.
            if part[0][self.k] < 0:
                return 0.
            res = 1
            #print ('part[1]',part[1])
            for l in range(part[1].shape[0]):
                #print (part[1].shape[0])
                kappa = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(part[0][self.k]*kappa) / st.beta(succ_l+1, fail_l+1).pdf(part_prev[0][self.k]*kappa)
            return res
        else:
            if part[1][self.k] > 1:
                return 0.
            if part[1][self.k] < 0:
                return 0.
            res = 1
            for l in range(part[0].shape[0]):
                theta = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(part[1][self.k]*theta) / st.beta(succ_l+1, fail_l+1).pdf(part_prev[1][self.k]*theta)
            return res
        
        
    def log_compute_rho(self, part_prev, part):
        """

        :param part_prev:
        :param part:
        :return:


        >>> import numpy as np
        >>> import scipy as sp
        >>> part1 = [np.array([0.8, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part2 = [np.array([0.5, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part3 = [np.array([-0.1, 0.5]), np.array([1, 0.8, 0.2])]
        >>> part4 = [np.array([1.1, 0.5]), np.array([1, 0.8, 0.2])]
        >>> target = Target({"success":[[8, 6, 2], [8, 8, 2]], "fail":[[2, 4, 8], [2, 4, 8]]}, k=0, on_theta=True)

        >>> target.log_compute_rho(part1, part1)
        0.0

        >>> true_val = sp.stats.beta(8+1, 2+1).logpdf(0.5*1) + sp.stats.beta(6+1, 4+1).logpdf(0.5*0.8) + sp.stats.beta(2+1, 8+1).logpdf(0.5*0.2) - sp.stats.beta(8+1, 2+1).logpdf(0.8*1) - sp.stats.beta(6+1, 4+1).logpdf(0.8*0.8) - sp.stats.beta(2+1, 8+1).logpdf(0.8*0.2)
        >>> val = target.log_compute_rho(part1, part2)
        >>> abs(val - true_val) < 1e-6
        True

        >>> val_bis = target.log_compute_rho(part2, part1)
        >>> abs(val + val_bis) < 1e-6
        True

        >>> target.log_compute_rho(part1, part3)
        -inf

        >>> target.log_compute_rho(part1, part4)
        -inf
        """
        if self.on_theta:
            if part[0][self.k] > 1:
                return - np.inf
            if part[0][self.k] < 0:
                return - np.inf
            res = 0
            #print ('part[1]',part[1])
            for l in range(part[1].shape[0]):
                #print (part[1].shape[0])
                kappa = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                x_new = part[0][self.k]*kappa
                res += sp.special.xlog1py(fail_l, -x_new) + sp.special.xlogy(succ_l, x_new)
                x_prev = part_prev[0][self.k]*kappa
                res -= sp.special.xlog1py(fail_l, -x_prev) + sp.special.xlogy(succ_l, x_prev)
            return res
        else:
            if part[1][self.k] > 1:
                return - np.inf
            if part[1][self.k] < 0:
                return - np.inf
            res = 0
            for l in range(part[0].shape[0]):
                theta = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                x_new = part[1][self.k]*theta
                res += sp.special.xlog1py(fail_l, -x_new) + sp.special.xlogy(succ_l, x_new)
                x_prev = part_prev[1][self.k]*theta
                res -= sp.special.xlog1py(fail_l, -x_prev) + sp.special.xlogy(succ_l, x_prev)
            return res
    
    def pdf_(self, part_prev):
        res = 1
        if self.on_theta:
            if part_prev[0][self.k] > 1:
                return 0.
            if part_prev[0][self.k] < 0:
                return 0.
            for l in range(part_prev[1].shape[0]):
                kappa_mono = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(part_prev[0][self.k ]*kappa_mono) * kappa_mono
            return res
        else:
            if part_prev[1][self.k] > 1:
                return 0.
            if part_prev[1][self.k] < 0:
                return 0.
            for l in range(part_prev[0].shape[0]):
               
                theta_mono = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res *= st.beta(succ_l+1, fail_l+1).pdf(theta_mono*part_prev[1][self.k]) * theta_mono
                #print (l, res)
            return res
        
    def log_pdf_(self, part_prev):
        res = 0
        if self.on_theta:
            if part_prev[0][self.k] > 1:
                return 0.
            if part_prev[0][self.k] < 0:
                return 0.
            for l in range(part_prev[1].shape[0]):
                kappa_mono = part_prev[1][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res += st.beta(succ_l+1, fail_l+1).logpdf(part_prev[0][self.k ]*kappa_mono) + math.log(kappa_mono)
            return res
        else:
            if part_prev[1][self.k] > 1:
                return 0.
            if part_prev[1][self.k] < 0:
                return 0.
            for l in range(part_prev[0].shape[0]):
                theta_mono = part_prev[0][l]
                succ_l = self.succ[l]
                fail_l = self.fail[l]
                res += st.beta(succ_l+1, fail_l+1).logpdf(theta_mono*part_prev[1][self.k]) + math.log(theta_mono) 
                #print (l, res)
            return res

       
    def pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                if part_prev[0][self.k] > 1:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                if part_prev[0][self.k] < 0:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                else :
                    res = 1
                    for l in range(part_prev[1].shape[0]):
                        #print ('l',l)
                        #print ('part_prev[1]',part_prev[1])
                        kappa_mono = part_prev[1][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res *= st.beta(succ_l+1, fail_l+1).pdf(part_prev[0][self.k]*kappa_mono) * kappa_mono
                    #print ('res',res)
                    x.append(part_prev[0][self.k])
                    y.append(res)
                
        else:
            for part_prev in part_prev_list:
                if part_prev[1][self.k] > 1:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                if part_prev[1][self.k] < 0:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                else:
                    res = 1
                    for l in range(part_prev[0].shape[0]):
                        theta_mono = part_prev[0][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res *= st.beta(succ_l+1, fail_l+1).pdf(theta_mono*part_prev[1][self.k]) * theta_mono
                    x.append(part_prev[1][self.k])
                    y.append(res)
        #print ('x',x,'y',y)
        return x,y
    
    def log_pdf_multiparticule(self, part_prev_list): 
        x = []
        y = []
        if self.on_theta: 
            for part_prev in part_prev_list:
                if part_prev[0][self.k] > 1:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                if part_prev[0][self.k] < 0:
                    x.append(part_prev[0][self.k])
                    y.append(0)
                else :
                    res = 0
                    for l in range(part_prev[1].shape[0]):
                        #print ('l',l)
                        #print ('part_prev[1]',part_prev[1])
                        kappa_mono = part_prev[1][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res += st.beta(succ_l+1, fail_l+1).logpdf(part_prev[0][self.k]*kappa_mono) + math.log(kappa_mono)
                    #print ('res',res)
                    x.append(part_prev[0][self.k])
                    y.append(res)
                
        else:
            for part_prev in part_prev_list:
                if part_prev[1][self.k] > 1:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                if part_prev[1][self.k] < 0:
                    x.append(part_prev[1][self.k])
                    y.append(0)
                else:
                    res = 0
                    for l in range(part_prev[0].shape[0]):
                        theta_mono = part_prev[0][l]
                        succ_l = self.succ[l]
                        fail_l = self.fail[l]
                        res += st.beta(succ_l+1, fail_l+1).logpdf(theta_mono*part_prev[1][self.k]) + math.log(theta_mono)
                    x.append(part_prev[1][self.k])
                    y.append(res)
        #print ('x',x,'y',y)
        return x,y


if __name__ == "__main__":
    import doctest

    doctest.testmod()
