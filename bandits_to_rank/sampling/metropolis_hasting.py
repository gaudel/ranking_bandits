#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Metropolis Hasting Algorithm"""

# Packages

from __future__ import division
import math
from bandits_to_rank.data.Methode_Simulation_KappasThetasKnown import *


"""Algorithme"""

def Metro_hast(proposals, post_ratios_list, part0, niters, Efficiency_show=False):
    
    naccept = np.zeros(part0[0].shape[0]+part0[1].shape[0])
    part = part0
    samples = [part]
        
    for i in range(niters):
        num_proposal = 0 ### donne la position de la proposal en jeu 
        for proposal, post_ratios in zip(proposals, post_ratios_list):
            part_next = proposal.next_part(part)
            #print ('part_next',part_next)
            #print ('part',part)
            rho_2 = proposal.compute_rho(part, part_next)
            ####/!\
            rho_1 = post_ratios.compute_rho(part, part_next)
            ####
            ##rho_1 = 1
            rho = min(1, rho_1*rho_2 )
            u = np.random.uniform()
            if u < rho:
                naccept[num_proposal] += 1
                part = part_next
            num_proposal +=1
        samples.append(part)
    
    efficiency = naccept/niters
    if Efficiency_show :
        print (efficiency)
    return samples


def log_Metro_hast(proposals, post_ratios_list, part0, niters, Efficiency_show=False):
    
    naccept = np.zeros(part0[0].shape[0]+part0[1].shape[0])
    part = part0
    samples = [part]
    reject_time_MH = 0
    for i in range(niters):
        num_proposal = 0
        
        for proposal, post_ratios in zip(proposals, post_ratios_list):
            part_next = proposal.next_part(part)
            reject_time_MH += proposal.reject_time_MA
            #print ('part_next',part_next)
            #print ('part',part)
            log_rho_2 = proposal.log_compute_rho(part, part_next)
            ####/!\
            log_rho_1 = post_ratios.log_compute_rho(part, part_next)
            ####
            ##rho_1 = 1
            log_rho = min(0, log_rho_1+log_rho_2 )
            log_u = math.log(np.random.uniform())
            #print ('threshold',log_rho_1+log_rho_2)
            #print (log_u)
            if log_u < log_rho:
                naccept[num_proposal] += 1
                part = part_next
            num_proposal +=1
        samples.append(part)
    
    efficiency = naccept/niters
    if Efficiency_show :
        return samples,efficiency,reject_time_MH
    return samples



"""Evaluation/Affichage"""

def split_sample (sample):
    nb_theta = len(sample[0][0])
    thetas =[[] for i in range(nb_theta)]
    nb_kappa = len(sample[0][1])
    kappas =[[] for i in range(nb_kappa)]
    for part in sample:
        for i in range(nb_theta):
            thetas[i].append(part[0][i])
    for part in sample:
        for i in range(nb_kappa):
            kappas[i].append(part[1][i]) 
    return thetas, kappas



if __name__ == "__main__":
    import doctest

    doctest.testmod()
