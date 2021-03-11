
#### Bandits

## Packages

import numpy as np
import random as rd

from bandits_to_rank.sampling.pbm_inference import *
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index

from pyclick.click_models.PBM import PBM
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult

from random import sample
from random import random




def extract_kappa(clickmodel,nb_position):
    param =[]
    for i,j in enumerate(clickmodel.params) :
        param.append(clickmodel.params[j])
    kappa=[]
    for i in range(nb_position):
        kappa.append(param[1]._container[i].value())
    return kappa

def give_kappa_Pyclick(sessions,nb_position):
    click_model = PBM()
    click_model.train(sessions)
    return extract_kappa (click_model,nb_position)

def extract_theta(clickmodel):
    param =[]
    for i,j in enumerate(clickmodel.params) :
        param.append(clickmodel.params[j])
    thetas ={}
    for i in param[0]._container['Reco']:
        thetas[i]= round(param[0]._container['Reco'][i].value(),4)
    return thetas

def order_thetas(thetas):
    return sorted(thetas.items(), key=lambda t: t[1], reverse=True)


def give_thetas_Pyclick(sessions):
    click_model = PBM()
    click_model.train(sessions)
    return order_thetas(extract_theta (click_model))


### General E_greedy
class EGreedy (object):
    def __init__(self, c, count_update, get_model):
        """

        Parameters
        ----------
        c : float
            explore with probability `c/t`
        count_update : int
            model is update any `_update` trials
        get_model : function Void -> Model
            to get a model to infer theta/kappa parameters
        """
        self.count_update = count_update
        self.c = c

        self.get_model = get_model

        self.clean()

        self.rng = np.random.default_rng()

    def clean(self):
        self.nb_trials = 0
        self.pbm_model = self.get_model()
        self.time_reject = 0

    def choose_next_arm(self, temps_init=10**(-5)):
        t = self.nb_trials + temps_init
        if self.rng.random() < self.c/t or self.nb_trials == 0:
            # explore
            return self.rng.choice(self.pbm_model.get_nb_products(), self.pbm_model.get_nb_positions(), replace=False), self.time_reject
        else:
            # exploit
            return order_theta_according_to_kappa_index(self.thetas_pyclic, self.kappas_pyclic), self.time_reject

    def update(self, propositions, rewards):
        self.nb_trials += 1
        self.pbm_model.add_session(propositions, rewards)
        if self.nb_trials <= 100 or self.nb_trials % self.count_update == 0:
            self.pbm_model.learn()
            self.thetas_pyclic, self.kappas_pyclic = self.pbm_model.get_params()

    def get_param_estimation(self):
        return self.thetas_pyclic, self.kappas_pyclic


### Specific E_greedy
# Wrap the function `get_model()` in a class to enable pickling
class GetSVD():
    def __init__(self, nb_arms, nb_positions):
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions

    def __call__(self):
        res = SVD(self.nb_arms, self.nb_positions)
        res.nb_views = np.ones((self.nb_arms, self.nb_positions), dtype=np.int) * 0.0001
        return res

def greedy_EGreedy(c, nb_arms, nb_position, count_update):
    """
    Epsilon Greedy algorithm based on rank-1 model (equivalent to PBM), with parameters inferred through SVD of empirical click-probabilities.
    """
    return EGreedy(c, count_update, GetSVD(nb_arms, nb_position))


# Wrap the function `get_model()` in a class to enable pickling
class GetEM():
    def __init__(self, nb_arms, nb_positions):
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions

    def __call__(self):
        res = EM(self.nb_arms, self.nb_positions)
        res.nb_views = np.ones((self.nb_arms, self.nb_positions), dtype=np.int) * 0.0001
        return res

def greedy_EGreedy_EM(c, nb_arms, nb_position, count_update):
    """
    Epsilon Greedy algorithm based on PBM, with parameters inferred through EM (pyclick version).
    """
    return EGreedy(c, count_update, GetEM(nb_arms, nb_position))
