#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

import os
import numpy as np
from math import log, exp

### Help func for referee
def build_scale(nb_trial, len_record_short=1000):
    """

    :param nb_trial:
    :param len_record_short:
    :return:

    >>> times = build_scale(1000, len_record_short=10)
    >>> print(times)
    [0, 2, 4, 8, 16, 32, 63, 100, 126, 200, 251, 300, 400, 500, 501, 600, 700, 800, 900, 999]

    """
    ### Equidist_part
    save_time_equidist = [i for i in range(0, nb_trial, int(nb_trial / len_record_short))] + [nb_trial - 1]
    ### Log part
    alpha = exp(log(nb_trial) / len_record_short)
    save_time_log = [int(round(alpha ** i, 0)) for i in range(1, len_record_short)]

    ### Union
    save_time = list(set().union(save_time_log, save_time_equidist))
    save_time.sort()
    save_time_final = [i for i in save_time if i < nb_trial]
    return save_time_final


def time_sec_to_DHMS(sec):
    day = int(sec // (3600 * 24))
    sec = sec % (3600 * 24)

    hour = int(sec // 3600)
    sec = sec % 3600

    minute = int(sec // 60)
    sec = sec % 60

    return (str(day) + 'day ' + str(hour) + 'h ' + str(minute) + 'min ' + str(sec) + 'sec')



## Fonction Auxiliaires


def maximum_K_index(liste, K=3):
    new=np.argsort(liste)
    return new[-1:-(int(K)+1):-1]

def maximum_K(liste,K=3):
    new=np.sort(liste)
    return new[-1:-(int(K)+1):-1]

def order_theta_according_to_kappa_index(thetas, kappas):
    """

    :param thetas:
    :param nb_position:
    :param kappas:
    :return:

    >>> import numpy as np
    >>> thetas = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.4])
    >>> kappas = np.array([1, 0.9, 0.8])
    >>> propositions = order_theta_according_to_kappa_index(thetas, 2, kappas)
    >>> assert(np.all(propositions == np.array([0, 1])), str(propositions))
    >>> propositions = order_theta_according_to_kappa_index(thetas, 3, kappas)
    >>> assert(np.all(propositions == np.array([0, 1, 2])), str(propositions))

    >>> thetas = np.array([0.6, 0.6, 0.8, 0.9, 0.7, 0.4])
    >>> kappas = np.array([1, 0.8, 0.9])
    >>> propositions = order_theta_according_to_kappa_index(thetas, 2, kappas)
    >>> assert(np.all(propositions == np.array([3, 4])), str(propositions))
    >>> propositions = order_theta_according_to_kappa_index(thetas, 3, kappas)
    >>> assert(np.all(propositions == np.array([3, 4, 2])), str(propositions))
    """
    nb_position = len(kappas)
    index_theta_ordonne = np.array(thetas).argsort()[::-1][:nb_position]
    index_kappa_ordonne =  np.array(kappas).argsort()[::-1][:nb_position]
    res = np.ones(nb_position, dtype=np.int)
    nb_put_in_res = 0
    for i in index_kappa_ordonne:
        res[i]=index_theta_ordonne[nb_put_in_res]
        nb_put_in_res+=1
    return res



def order_index_according_to_kappa(index, kappas):
    """

    :param index:
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
    index_kappa_order = np.array(kappas).argsort()[::-1][:nb_position]
    res = np.ones(nb_position, dtype=np.int)
    nb_put_in_res = 0
    for i in index_kappa_order:
        res[i]=index[nb_put_in_res]
        nb_put_in_res+=1
    return res



if __name__ == "__main__":
    import doctest

    doctest.testmod()
