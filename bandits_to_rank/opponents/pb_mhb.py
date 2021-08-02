
#### Bandits

## Packages


import numpy as np
import random as rd


from bandits_to_rank.sampling.metropolis_hasting import *
from bandits_to_rank.sampling.proposal import *
from bandits_to_rank.sampling.target import *
from bandits_to_rank.opponents.greedy import GetSVD
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index

from numpy.random import beta 
from random import uniform
from copy import deepcopy


#from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import partial

### get proposals 
      
class propos_trunk_GRW():
    
    def __init__(self,c,vari_sigma):
        self.vari_sigma = vari_sigma
        self.c = c
        self.turn = 0


    def set_arms_and_positions(self, nb_arms, nb_position):
        self.nb_arms = nb_arms
        self.nb_position = nb_position

            
    def update_parampropose(self, success, fail):
        self.turn +=1
        if self.vari_sigma :
            self.sigma = (self.c / math.sqrt(self.turn))
        else:
            self.sigma = self.c
                  
    
    def get_proposal(self):
        return [TruncatedGaussianRandomWalk(on_theta=True,sigma = self.sigma, k=i) for i in range(self.nb_arms)] + [TruncatedGaussianRandomWalk(on_theta=False,sigma = self.sigma, k=i) for i in range(1, self.nb_position)]
    
    

        
class propos_RW():
    def __init__(self,c,vari_sigma):
        self.vari_sigma = vari_sigma
        self.c = c
        self.turn = 0
       
    def set_arms_and_positions(self, nb_arms, nb_position):
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        
    def update_parampropose(self, success, fail):
        self.turn +=1
        if self.vari_sigma :
            self.sigma=(self.c / math.sqrt(self.turn))
        else:
            self.sigma = self.c

    
    def get_proposal(self):
        return [Proposal_RW(on_theta=True, sigma = self.sigma, k=i) for i in range(self.nb_arms)] + [Proposal_RW(on_theta=False, sigma = self.sigma, k=i) for i in range(1, self.nb_position)]

        
class propos_logit_RW():
    
    def __init__(self,c,vari_sigma):
        self.vari_sigma = vari_sigma
        self.c = c
        self.turn = 0
       
    def set_arms_and_positions(self, nb_arms, nb_position):
        self.nb_arms = nb_arms
        self.nb_position = nb_position

    def update_parampropose(self, success, fail):
        self.turn+=1
        if self.vari_sigma:
            self.sigma = (self.c / math.sqrt(self.turn))
        else:
            self.sigma = self.c
    
    def get_proposal(self):
        return [Proposal_RW_logit(on_theta=True, sigma = self.sigma, k=i) for i in range(self.nb_arms)] + [Proposal_RW_logit(on_theta=False, sigma = self.sigma, k=i) for i in range(1, self.nb_position)]
    
    
    
class propos_max_position():
    
    def __init__(self):
        pass

    def set_arms_and_positions(self, nb_arms, nb_position):
        self.nb_arms = nb_arms
        self.nb_position = nb_position
    
    def build_dico(self, success, fail):
        dico={}
        dico['success']=success
        dico['fail']=fail
        nb_arm = success.shape[0]
        seen = success+fail
        dico['most_seen'] = self.build_list_most_coupled_with(seen)
        dico['most_placed'] = self.build_list_most_coupled_with(seen,based_on_item=False)
        return dico
        
    def build_list_most_coupled_with (self, seen, based_on_item = True):
        """
        Construct an array of the places on which each product was the
        most placed. ex:most_coupled[i]= l means that the product i was mostly placed in position l until now, making l the best partner of i when display, if based_on_item = True
        otherwise, we are looking for the product placed the most often at a given place. ex:most_coupled[l]= i means that the place l was mostly filed with the product i until now
        """
        if based_on_item:
            matrix_time_coupled = seen
        else:
            matrix_time_coupled =np.transpose(seen)
            
        most_coupled = []
        for i in range(len(matrix_time_coupled)):
            index_best_partner = np.argmax(matrix_time_coupled[i])
            most_coupled.append(index_best_partner)
        return most_coupled
  
        
    def update_parampropose(self, success, fail):
         self.dico = self.build_dico(success, fail)
                  
    
    def get_proposal(self):
        return [Proposal_maxposition(on_theta=True,dico = self.dico, k=i) for i in range(self.nb_arms)] + [Proposal_maxposition(on_theta=False,dico = self.dico, k=i) for i in range(1, self.nb_position)]
    
    

    
class propos_pseudo_view():
    
    def __init__(self):
        pass

    def set_arms_and_positions(self, nb_arms, nb_position):
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        
    def build_dico(self, success, fail):
        dico={}
        dico['success']=success
        dico['fail']=fail
        nb_arm = success.shape[0]
        seen = success+fail
        dico['most_seen'] = self.build_list_most_coupled_with(seen)
        dico['most_placed'] = self.build_list_most_coupled_with(seen,based_on_item=False)
        return dico
        
    def build_list_most_coupled_with (self, seen, based_on_item = True):
        """
        Construct an array of the places on which each product was the
        most placed. ex:most_coupled[i]= l means that the product i was mostly placed in position l until now, making l the best partner of i when display, if based_on_item = True
        otherwise, we are looking for the product placed the most often at a given place. ex:most_coupled[l]= i means that the place l was mostly filed with the product i until now
        """
        if based_on_item:
            matrix_time_coupled = seen
        else :
            matrix_time_coupled = np.transpose(seen)
            
        most_coupled = []
        for i in range(len(matrix_time_coupled)):
            index_best_partner = np.argmax(matrix_time_coupled[i])
            most_coupled.append(index_best_partner)
        return most_coupled
  
        
    def update_parampropose(self, success, fail):
         self.dico = self.build_dico(success, fail)
                  
    
    def get_proposal(self):
        return [Proposal_pseudoViewBis(on_theta=True, dico = self.dico, k=i) for i in range(self.nb_arms)] + [Proposal_pseudoViewBis(on_theta=False, dico = self.dico, k=i) for i in range(1, self.nb_position)]


class propos_Round_Robin():

    def __init__(self,c,vari_sigma,list_proposal_possible):
        self.vari_sigma = vari_sigma
        self.c = c
        self.list_proposal_possible =list_proposal_possible
        self.nb_proposal_possible = len(list_proposal_possible)
        self.proposal_type_at_this_turn = list_proposal_possible[0]
        self.turn = 0

    def set_arms_and_positions(self, nb_arms, nb_position):
        self.nb_arms = nb_arms
        self.nb_position = nb_position

    def update_parampropose(self, success, fail):
        self.turn += 1
        for i in range(self.nb_proposal_possible):
            if self.turn%self.nb_proposal_possible == i:
                self.proposal_type_at_this_turn = self.list_proposal_possible[i]

        if self.vari_sigma:
            self.sigma = (self.c / math.sqrt(self.turn))
        else:
            self.sigma = self.c

        self.dico = self.build_dico(success, fail)

    def update_parampropose_old(self, success, fail):
        self.turn += 1
        if self.turn%3 == 1:
            self.proposal_type_at_this_turn = 'TGRW'
        #elif self.turn%4 == 2:
        #    self.proposal_type_at_this_turn = 'LGRW'
        elif self.turn%3 == 2:
            self.proposal_type_at_this_turn = 'Pseudo_View'
        else:
            self.proposal_type_at_this_turn = 'Max_Position'

        if self.vari_sigma:
            self.sigma = (self.c / math.sqrt(self.turn))
        else:
            self.sigma = self.c

        self.dico = self.build_dico(success, fail)

    def build_dico(self, success, fail):
        dico = {}
        dico['success'] = success
        dico['fail'] = fail
        nb_arm = success.shape[0]
        seen = success + fail
        dico['most_seen'] = self.build_list_most_coupled_with(seen)
        dico['most_placed'] = self.build_list_most_coupled_with(seen, based_on_item=False)
        return dico

    def build_list_most_coupled_with(self, seen, based_on_item=True):
        """
        Construct an array of the places on which each product was the
        most placed. ex:most_coupled[i]= l means that the product i was mostly placed in position l until now, making l the best partner of i when display, if based_on_item = True
        otherwise, we are looking for the product placed the most often at a given place. ex:most_coupled[l]= i means that the place l was mostly filed with the product i until now
        """
        if based_on_item:
            matrix_time_coupled = seen
        else:
            matrix_time_coupled = np.transpose(seen)

        most_coupled = []
        for i in range(len(matrix_time_coupled)):
            index_best_partner = np.argmax(matrix_time_coupled[i])
            most_coupled.append(index_best_partner)
        return most_coupled


    def get_proposal(self):
        if self.proposal_type_at_this_turn == 'TGRW':
            return [TruncatedGaussianRandomWalk(on_theta=True, sigma=self.sigma, k=i) for i in range(self.nb_arms)] + [
            TruncatedGaussianRandomWalk(on_theta=False, sigma=self.sigma, k=i) for i in range(self.nb_position)]

        elif self.proposal_type_at_this_turn == 'LGRW':
            return [Proposal_RW_logit(on_theta=True, sigma=self.sigma, k=i) for i in range(self.nb_arms)] + [
                Proposal_RW_logit(on_theta=False, sigma=self.sigma, k=i) for i in range(self.nb_position)]

        elif self.proposal_type_at_this_turn == 'Pseudo_View':
            return [Proposal_pseudoViewBis(on_theta=True, dico=self.dico, k=i) for i in range(self.nb_arms)] + [
                Proposal_pseudoViewBis(on_theta=False, dico=self.dico, k=i) for i in range(self.nb_position)]

        elif self.proposal_type_at_this_turn == 'Max_Position':
            return [Proposal_maxposition(on_theta=True, dico=self.dico, k=i) for i in range(self.nb_arms)] + [
                Proposal_maxposition(on_theta=False, dico=self.dico, k=i) for i in range(self.nb_position)]

        else :
            raise ValueError(
                f'{self.proposal_type_at_this_turn} is not an implemented proposal.')


## TS_MH

    
class PB_MHB:
    """
     C-S. Gauthier, R. Gaudel, E. Fromont
    Position-Based Multiple-Play Bandits with Thompson Sampling
    PB-MHB
    """
    def __init__(self, nb_arms, nb_position, proposal_method=propos_trunk_GRW(vari_sigma=True, c=3), initial_particule=None, step=10, prior_s=0.5, prior_f=0.5, part_followed=True,  store_eff=False):
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.step = step
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.initial_particule = initial_particule
        self.part_followed = part_followed
        self.positions = np.arange(nb_position)
        self.store_eff = store_eff
        self.proposal_method = proposal_method
        self.proposal_method.set_arms_and_positions(nb_arms, nb_position)
        self.get_model= GetSVD(self.nb_arms, self.nb_position)
        self.clean()

    def _random_particule(self):
        return [np.random.uniform(0, 1, self.nb_arms),
                np.array([1] + list(np.random.uniform(0, 1, self.nb_position - 1)))]

    def clean(self):
        """ Clean log data.
        To be ran before playing a new game.
        """
        self.success = np.ones([self.nb_arms, self.nb_position], dtype=np.uint)*self.prior_s
        self.fail = np.ones([self.nb_arms, self.nb_position], dtype=np.uint)*self.prior_f
        if self.initial_particule is not None:
            self.particule = deepcopy(self.initial_particule)
        else:
            self.particule = self._random_particule()
        if self.store_eff:
            self.eff = []
        self.turn = 1
        self.reject_time = 0
        self.pbm_model = self.get_model()

        


    def choose_next_arm(self):
        ### Build the target
        targets = [Target_TS(self.success, self.fail, i) for i in range(self.nb_arms)] + [Target_TS(self.success, self.fail, k=i,on_theta=False) for i in range(1,self.nb_position)]
        
        ### Build the proposal
        self.proposal_method.update_parampropose(self.success, self.fail)
        proposals = self.proposal_method.get_proposal()
        
        ### Compute MH
        samples, eff, reject_time = log_Metro_hast (proposals, targets, self.particule, self.step, True)
        self.reject_time = reject_time
        if self.part_followed:
            self.particule = samples[-1]
        else :
            self.particule = self._random_particule()
        if self.store_eff:
            self.eff.append((self.step - (eff * self.step)) / self.step)
            
        thetas = samples[-1][0]
        kappas = samples[-1][1]
        return order_theta_according_to_kappa_index(thetas, kappas), reject_time
    
    def update(self, propositions, rewards):
        self.turn +=1
        self.fail[propositions, self.positions] += 1 - rewards
        self.success[propositions, self.positions] += rewards
        self.pbm_model.add_session(propositions, rewards)

    def get_param_estimation(self):
        self.pbm_model.learn()
        self.thetas_estim, self.kappas_estim = self.pbm_model.get_params()
        return self.thetas_estim, self.kappas_estim


    def type(self):
        return 'PB-MHB'
    

  