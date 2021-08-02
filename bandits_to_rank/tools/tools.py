#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

import os
import numpy as np
from math import log, exp
from scipy.optimize import fminbound, root_scalar

def get_SCRATCHDIR():
    if "SCRATCHDIR" in os.environ.keys():
        return os.environ["SCRATCHDIR"]
    else:
        return "."


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



## Manipulating permutations

def swap(permutation, transposition, remaining=[]):
    """ Composition of a permutation with a transposition.
    Swap two entries in the permutation. The second entry may be outside of the permutation (and taken from `remaining` elements)

    Parameters
    ----------
    permutation: iterable of size N (won't be changed)
        The permutation
    transposition: tuple of size two
        The indexes of both entries to swap. The second index `j` may be greater than `N`, which means the entry `j-N` in remaining will be used instead.
    remaining: iterable
        Values to be used if the second index is greater than the size of th permutation.

    Returns
    -------
    new_permutation: tuple of size N

    Examples
    --------
    >>> swap((1,4,0), (0,0), (2, 3, 5))
    (1, 4, 0)
    >>> swap((1,4,0), (0,1), (2, 3, 5))
    (4, 1, 0)
    >>> swap((1,4,0), (1,2), (2, 3, 5))
    (1, 0, 4)
    >>> swap((1,4,0), (2,3), (2, 3, 5))
    (1, 4, 2)

    >>> perm = np.array((1, 4, 0))
    >>> remaining = np.array((2, 3, 5))
    >>> swap(perm, (2,3), (2, 3, 5))
    (1, 4, 2)
    >>> perm
    array([1, 4, 0])
    >>> swap(perm, (2,3), remaining)
    (1, 4, 2)
    >>> perm
    array([1, 4, 0])
    >>> remaining
    array([2, 3, 5])
    """
    i, j = transposition
    nb_positions = len(permutation)
    res = np.array(permutation)

    if j < nb_positions:
        res[i], res[j] = res[j], res[i]
    else:
        res[i] = remaining[j-nb_positions]

    return tuple(res)


def swap_full(permutation, transposition,nb_position):
    """ Composition of a permutation with a transposition.
    Swap two entries in the permutation and sized it according to nb_position
    Parameters
    ----------
    permutation: iterable of size N
        The permutation
    transposition: tuple of size two
        The indexes of both entries to swap.

    nb_position: size of the final results

    Returns
    -------
    new_permutation: tuple of size nb_position

    Examples
    --------
    >>> swap((1,4,0,2,3,5), (0,0), 3)
    (1, 4, 0)
    >>> swap((1,4,0,2,3,5), (0,1), 3)
    (4, 1, 0)
    >>> swap((1,4,0,2,3,5), (1,2), 3)
    (1, 0, 4)
    >>> swap((1,4,0,2,3,5), (2,3), 3)
    (1, 4, 2)

    >>> perm = np.array((1, 4, 0, 2, 3, 5))
    >>> nb_position = 3
    >>> swap(perm, (2,3), nb_position)
    (1, 4, 2)
    >>> perm
    array([1, 4, 0])
    """
    i, j = transposition
    res = np.array(permutation)
    res[i], res[j] = res[j], res[i]
    return tuple(res[:nb_position])

def unused(permutation, nb_elements):
    """ List the elements of `range(nb_elements)` which are not in `permutation`

    Parameters
    ----------
    permutation: iterable
    nb_elements: int

    Returns
    -------
    unused_elements: tuple

    Examples
    --------
    >>> unused((1, 4, 0), 6)
    (2, 3, 5)
    """
    return tuple(set(range(nb_elements)) - set(permutation))

#### KL UCB
def kullback_leibler_divergence(p, q):
    # print(p,q)
    return p * log(p / q) + (1 - p) * log(abs((1 - p) / (1 - q)))


def bound_KL_brentq(mu, certitude, n):
    mu_min = fminbound(kullback_leibler_divergence, 0, 1, args=[mu])
    mu_max = 1 - 1 / n
    sol = root_scalar((lambda x: (kullback_leibler_divergence(mu, x) - certitude / n)), bracket=[mu_min, mu_max],
                      method='brentq')
    return sol.root


def newton(mu, certitude, n, x0, tol=1.48e-8, maxiter=50):
    #C'est une légère modification de la méthode de newton codée dans scipy.optimize
    #mu, certitude et n vont définir f et fprime:
    #certitude = -log(p) où p est la confiance de l'intervalle de confiance que nous voulons définir
    # f = lambda x: mu * log(mu / x) + (1 - mu) * log((1 - mu) / (1 - x)) - certitude / n
    #fprime = lambda x: (x - mu) / (x * (1 - x))
    p0 = 1.0 * x0
    if mu == 0:
        for itr in range(maxiter):
        # first evaluate fval
            fval = - log(1 - p0) - certitude / n
        # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return p0
            newton_step = fval *(1 - p0)
            p = p0 - newton_step
            if abs(newton_step) <= tol:
                return p
            p0 = p
    elif mu == 1:
        return 1
    else:
        for itr in range(maxiter):
        # first evaluate fval
            fval = mu * log(mu / p0) + (1 - mu) * log((1 - mu) / (1 - p0)) - certitude / n
        # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return p0
            newton_step = (p0 * (1 - p0)) * fval / (p0 - mu)
            p = p0 - newton_step
            if abs(newton_step) <= tol:
                return p
            p0 = p

def start_up(mu, certitude, n):
    #mu, certitude et n servent à définir f
    # f = lambda x: mu * log(mu / x) + (1 - mu) * log((1 - mu) / (1 - x)) - certitude / n

    # On cherche k tel que f(r_k) < 0 et f(r_k+1) > 0 et renvoie r_k où r_0 = (1 + mu)/ 2 et 1 - r_k+1 = (1 - r_k)/10
    # Voir la courbe de f (i.e de KL)
    # mu est le point où f atteint son minimum
    res = (1 + mu) / 2
    if mu == 0:
        while (- log(1 - res) - certitude / n < 0):
            next_res = 1 - (1 - res) / 10
            if next_res == 1: #a cause des erreurs d'approximation des réels
                return res
            res = next_res
        return res
    elif mu == 1:
        return 1
    else:
        while (mu * log(mu / res) + (1 - mu) * log((1 - mu) / (1 - res)) - certitude / n < 0):
            next_res = 1 - (1 - res) / 10
            if next_res == 1: #a cause des erreurs d'approximation des réels
                return res
            res = next_res
        return res
### Order proposition

def maximum_K_index(liste, K=3):
    new = np.argsort(liste)
    return new[-1:-(int(K)+1):-1]


def maximum_K(liste,K=3):
    new = np.sort(liste)
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
