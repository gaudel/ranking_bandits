#### Package
from random import gauss
from random import randint
from random import sample
from scipy.stats import bernoulli
from random import random

import numpy as np



def simule_log(run,theta,kappa):
    log={}
    log['session']=[]
    log['presentation']=[]
    nb_position = len(kappa)
    nb_item = len(theta)
    mat_position = np.zeros([nb_item,nb_position])
    for reco in range(run):
        ### draw randomly products presentation:
        l = [x for x in range(nb_item)]
        prod_presentation = sample(l,nb_position)
        log['presentation'].append(prod_presentation)
        for pos in range(nb_position):
            mat_position[prod_presentation[pos]][pos] +=1

        ###simulation behaviour
        session = []
        for pos in range(nb_position):
            index_product = prod_presentation[pos]
            is_view = int(random() < kappa[pos])
            is_click = int(random() < theta[index_product])
            session.append(is_view*is_click)
        log['session'].append(session)
    return(log, mat_position)


def perf_product(log, theta, kappa):
    nb_product = len(theta)
    nb_place = len(kappa)
    sucess = np.zeros([nb_product,nb_place])
    fail = np.zeros([nb_product,nb_place])
    for session in range(len(log['session'])):
        for position in range(len(log['session'][1])):
            item = log['presentation'][session][position]
            if log['session'][session][position] == 1 :
                
                sucess[item][position] += 1
            else:
                fail[item][position] += 1
    return (sucess, fail)



def max_position_put(mat):
    l,c = np.shape(mat)
    max_posmis_par_item = np.zeros(l)
    for i in range(l):
        max_posmis_par_item[i] = np.argmax(mat[i])
    return max_posmis_par_item


def product_most_put(mat):
    t_mat = np.transpose(mat)
    l,c = np.shape(t_mat)
    product_most_put_per_position = np.zeros(l)
    for i in range(l):
        product_most_put_per_position[i] = np.argmax(t_mat[i])
    return product_most_put_per_position


def Simulation(nb_simul, theta, kappa):
    dic = {}
    dic['nb_simul'] = nb_simul
    dic['theta'] = theta
    dic['kappa'] = kappa
    
    # log simulation
    log,mat_position = simule_log(nb_simul,theta,kappa)
    dic['log'] = log
    dic['mat_position'] = mat_position
    
    # identify position where products where the most put
    max_seen = max_position_put(mat_position)
    dic['max_seen'] = max_seen
    
     # identify products the most put on each position
    prod_most_put = product_most_put(mat_position)
    dic['prod_most_put'] = prod_most_put
    
    # build fail and succes
    sucess, fail = perf_product(log, theta, kappa)
    dic['success'] = sucess
    dic['fail'] = fail
    dic['fail_compress'] = np.sum(fail,axis=0)
    dic['sucess_compress'] = np.sum(sucess,axis=0)
    
    return dic