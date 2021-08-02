
#### Bandits

## Packages

import numpy as np
import random as rd

from bandits_to_rank.sampling.pbm_inference import *
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index
from bandits_to_rank.tools.get_inference_model import GetSVD, GetEM, GetMLE

from pyclick.click_models.PBM import PBM
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult

from random import sample
from random import random



def simule_log_Pyclick(nb_reco,theta,kappa):
    search_sessions=[]
    nb_position = len(kappa)
    nb_item = len(theta)
    indice_item = [x for x in range(nb_item)]
    for reco in range(nb_reco):
        #print('recommandation numero = ',reco)
        web_results =[]
        ### tire aléatoirement la présentation des produits :
        proposition=sample(indice_item,nb_position)
        #print ('produits proposes',proposition)
        for pos in range(len(proposition)):
            #print ('a la position', pos )
            index_produit = proposition[pos]
            #print('je propose',index_produit)
            is_view = int(random() < kappa[pos])
            is_click= int(random() < theta[index_produit])
            web_results.append(SearchResult(index_produit,is_view*is_click))
        ###simulation comportement
        search_sessions.append(TaskCentricSearchSession(reco,'Reco'))
        search_sessions[-1].web_results = web_results
    return(search_sessions)

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


## Algo

class greedy:
    """Construction d'un Bandit
    Source:
      Parameter
    nb_arms : nombre de bras a etudier
    kappas : cllick rate of prositions
      Attributs:
    performance = cumule des fois ou le bras a ete clique,
    nb_trials : array(n,2) nombre de fois ou les categories sont vues,
    nb_position
    """

    def __init__(self, nb_arms, nb_position, count_update):
        self.nb_trials = 0
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.count_update = count_update
        self.pbm_model = SVD(nb_arms, nb_position)
        self.thetas_pyclic =  np.zeros(nb_arms)
        self.kappas_pyclic =  np.zeros(nb_position)
        self.pbm_model.nb_views = np.ones((nb_arms, nb_position), dtype=np.int)*0.0001
        self.time_reject = 0

    def choose_next_arm(self):
        if self.nb_trials == 0 :
            self.nb_trials += 1
            return rd.sample(range(self.nb_arms),self.nb_position), self.time_reject
        else :
            if (self.nb_trials < 100) :
                self.pbm_model.learn()
                self.thetas_pyclic,self.kappas_pyclic =   self.pbm_model.get_params()
            else :
                if self.nb_trials%self.count_update == 0 :
                    self.pbm_model.learn()
                    self.thetas_pyclic,self.kappas_pyclic =   self.pbm_model.get_params()
            self.nb_trials += 1
            return order_theta_according_to_kappa_index(self.thetas_pyclic, self.kappas_pyclic), self.time_reject


    def update(self, propositions, rewards):
         self.pbm_model.add_session(propositions,rewards)


    def type(self):
        return 'greedy'


class greedy_EM:
    """Construction d'un Bandit
    Source:
      Parameter
    nb_arms : nombre de bras a etudier
    kappas : cllick rate of prositions
      Attributs:
    performance = cumule des fois ou le bras a ete clique,
    nb_trials : array(n,2) nombre de fois ou les categories sont vues,
    nb_position
    """

    def __init__(self, nb_arms, nb_position, count_update):
        self.nb_trials = 0
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.count_update = count_update
        self.pbm_model = EM(nb_arms, nb_position)
        self.thetas_pyclic =  np.zeros(nb_arms)
        self.kappas_pyclic =  np.zeros(nb_position)
        self.pbm_model.nb_views = np.ones((nb_arms, nb_position), dtype=np.int)*0.0001
        self.time_reject = 0

    def choose_next_arm(self):
        if self.nb_trials == 0 :
            self.nb_trials += 1
            return rd.sample(range(self.nb_arms),self.nb_position)
        else :
            if (self.nb_trials < 100) :
                self.pbm_model.learn()
                self.thetas_pyclic,self.kappas_pyclic =   self.pbm_model.get_params()
            else :
                if self.nb_trials%self.count_update == 0 :
                    self.pbm_model.learn()
                    self.thetas_pyclic,self.kappas_pyclic =   self.pbm_model.get_params()
            self.nb_trials += 1
            #if self.nb_trials%100 == 0  :
                #print ('Greedy trial n° '+str(self.nb_trials))
            return order_theta_according_to_kappa_index(self.thetas_pyclic, self.kappas_pyclic), self.time_reject


    def update(self, propositions, rewards):
         self.pbm_model.add_session(propositions,rewards)


    def type(self):
        return 'greedy_Cascade'




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



def greedy_EGreedy(c, nb_arms, nb_position, count_update):
    """
    Epsilon Greedy algorithm based on rank-1 model (equivalent to PBM), with parameters inferred through SVD of empirical click-probabilities.
    """
    return EGreedy(c, count_update, GetSVD(nb_arms, nb_position))



def greedy_EGreedy_EM(c, nb_arms, nb_position, count_update):
    """
    Epsilon Greedy algorithm based on PBM, with parameters inferred through EM (pyclick version).
    """
    return EGreedy(c, count_update, GetEM(nb_arms, nb_position))
