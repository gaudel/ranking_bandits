### Environment 

## Packages

from random import random
import random as rd
import numpy as np 

# from bandits import maximum_K_index, maximum_K
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index, maximum_K_index, maximum_K


### Helping Fonctions

## Environment

class Environment_PBM:
    """
    Describe the comportement of a user in front of a list of item
    Returns a list of rewards : r_k = 1  with probability  tehta_k  and 0 otherwise
    """

    def __init__(self, thetas, kappas, label=None):
        self.thetas = np.array(thetas)
        self.kappas = np.array(kappas)
        self.label = label
        self.rng = np.random.default_rng()

    def shuffle(self, fixed_kappa=False):
        """Shuffle items and positions

        first position is not shuffled


        >>> from bandits_to_rank.environment import Environment_PBM
        >>> import random
        >>> import numpy as np
        >>> np.set_printoptions(precision=2)

        >>> thetas = [0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
        >>> kappas = [1, 0.7, 0.5, 0.4, 0.3]
        >>> env = Environment_PBM(thetas, kappas)

        >>> env.get_best_index_decrease()
        array([0, 1, 2, 3, 4])
        >>> env.get_best_index()
        array([0, 1, 2, 3, 4])

        >>> random.seed(1)
        >>> env.shuffle(fixed_kappa=True)
        >>> env.thetas
        [0.8, 0.3, 0.9, 0.5, 0.2, 0.1, 0.4]
        >>> env.get_best_index_decrease()
        array([2, 0, 3, 6, 1])
        >>> env.kappas
        [1, 0.7, 0.5, 0.4, 0.3]
        >>> env.get_best_index()
        array([2, 0, 3, 6, 1])

        >>> env.shuffle()
        >>> env.thetas
        [0.5, 0.1, 0.4, 0.3, 0.8, 0.2, 0.9]
        >>> env.get_best_index_decrease()
        array([6, 4, 0, 2, 3])
        >>> env.kappas
        [1, 0.3, 0.5, 0.7, 0.4]
        >>> env.get_best_index()
        array([6, 3, 0, 4, 2])
        """
        self.rng.shuffle(self.thetas)
        if not fixed_kappa:
            self.rng.shuffle(self.kappas[1:])

    def get_reward(self, propositions):
        return np.array(self.rng.random() < self.thetas[propositions] * self.kappas, dtype=np.int)

    def _kappas(self):
        return self.kappas
    
    def _thetas(self):
        return self.thetas

    def get_setting(self):
        return len(self.thetas), len(self.kappas)

    # def get_best(self):
    #    return ordonne_theta_function_kappa(self.thetas,self.kappas)

    def get_best_index(self):
        return order_theta_according_to_kappa_index(self.thetas, self.kappas)

    def get_best_decrease(self):
        nb_position = len(self.kappas)
        return maximum_K(self.thetas, nb_position)

    def get_best_index_decrease(self):
        nb_position = len(self.kappas)
        return maximum_K_index(self.thetas, nb_position)

    def get_expected_reward(self, propositions):
        return self.kappas * self.thetas[propositions]

    def get_params(self):
        return {"label": self.label, "thetas": self.thetas, "kappas": self.kappas}


class Environment_multirequest_PBM:
    """
    Describe the comportement of a user in front of a list of item
    Returns a list of rewards : r_k = 1  with probability  tehta_k  and 0 otherwise
    """

    def __init__(self, thetas, kappas):
        self.thetas = thetas
        self.kappas = np.array(kappas)
        self.rng = np.random.default_rng()

    def shuffle(self, fixed_kappa=False):
        """Shuffle items and positions

        first position is not shuffled


        >>> from bandits_to_rank.environment import Environment_PBM
        >>> import random
        >>> import numpy as np
        >>> np.set_printoptions(precision=2)

        >>> thetas = {1:[0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1],
                      2:[0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
                      3:[0.19, 0.8, 0.35, 0.4, 0.23, 0.2, 0.61]}
        >>> kappas = [1, 0.7, 0.5, 0.4, 0.3]
        >>> env = Environment_multirequest_PBM(thetas, kappas)

        >>> env.get_best_index_decrease(1)
        array([0, 1, 2, 3, 4])
        >>> env.get_best_index(1)
        array([0, 1, 2, 3, 4])

        >>> random.seed(1)
        >>> env.shuffle(1,fixed_kappa=True)
        >>> env.thetas[1]
        [0.8, 0.3, 0.9, 0.5, 0.2, 0.1, 0.4]
        >>> env.get_best_index_decrease(1)
        array([2, 0, 3, 6, 1])
        >>> env.kappas
        [1, 0.7, 0.5, 0.4, 0.3]
        >>> env.get_best_index(1)
        array([2, 0, 3, 6, 1])

        >>> env.shuffle(1)
        >>> env.thetas
        [0.5, 0.1, 0.4, 0.3, 0.8, 0.2, 0.9]
        >>> env.get_best_index_decrease(1)
        array([6, 4, 0, 2, 3])
        >>> env.kappas
        [1, 0.3, 0.5, 0.7, 0.4]
        >>> env.get_best_index(1)
        array([6, 3, 0, 4, 2])
        """
        self.rng.shuffle(self.thetas)
        if not fixed_kappa:
            self.rng.shuffle(self.kappas[1:])

    def get_reward(self, propositions, query):
        return np.array(self.rng.random() < self.thetas[query][propositions] * self.kappas, dtype=np.int)

    def _kappas(self):
        return self.kappas

    def _thetas(self):
        return self.thetas

    def _thetas_query(self, query):
        return self.thetas[query]

    def _query_nb(self):
        return len(self.thetas.keys())

    def _query_list(self):
        return self.thetas.keys()

    def get_setting(self, query):
        return len(self.thetas[query]), len(self.kappas)

    def get_next_query(self):
        return rd.choice(list(self._query_list()))

    # def get_best(self):
    #    return ordonne_theta_function_kappa(self.thetas,self.kappas)

    def get_best_index(self, query):
        return order_theta_according_to_kappa_index(self.thetas[query], self.kappas)

    def get_best_decrease(self, query):
        nb_position = len(self.kappas)
        return maximum_K(self.thetas[query], nb_position)

    def get_best_index_decrease(self, query):
        nb_position = len(self.kappas)
        return maximum_K_index(self.thetas[query], nb_position)

    def get_expected_reward(self, propositions, query):
        return self.kappas * self.thetas[query][propositions]

    def get_params(self):
        return {"thetas": self.thetas, "kappas": self.kappas}


class Environment_KDD:
    """
    Describe the comportement of a user in front of a list of item
    Returns a list of rewards : r_k = 1  with probability  tehta_k  and 0 otherwise
    """
    def __init__(self, thetas, kappas):
        self.thetas = thetas
        self.kappas = kappas

    def shuffle(self, fixed_kappa=False):
        """Shuffle items and positions

        first position is not shuffled


        >>> from bandits_to_rank.environment import Environment_PBM
        >>> import random
        >>> import numpy as np
        >>> np.set_printoptions(precision=2)

        >>> thetas = [0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
        >>> kappas = [1, 0.7, 0.5, 0.4, 0.3]
        >>> env = Environment_PBM(thetas, kappas)

        >>> env.get_best_index_decrease()
        array([0, 1, 2, 3, 4])
        >>> env.get_best_index()
        array([0, 1, 2, 3, 4])

        >>> random.seed(1)
        >>> env.shuffle()
        >>> env.thetas
        [0.8, 0.3, 0.9, 0.5, 0.2, 0.1, 0.4]
        >>> env.get_best_index_decrease()
        array([2, 0, 3, 6, 1])
        >>> env.kappas
        [1, 0.3, 0.4, 0.5, 0.7]
        >>> env.get_best_index()
        array([2, 1, 6, 3, 0])
        """
        print("!Warning! Not implemented, environment_KDD is deprecated")

    def get_reward(self,propositions,query):
        rewards =[]
        for pos in range(len(propositions)) : 
            item = propositions[pos]
            rewards.append(int(random() < self.thetas[query][item]*self.kappas[query][pos]))
        return rewards

    def _kappas(self):
        return self.kappas
    
    def _thetas(self):
        return self.thetas
    
    def _kappas_query(self,query):
        return self.kappas[query]
    
    def _thetas_query(self,query):
        return self.thetas[query]
    
    def get_setting(self,query):
        return len(self.thetas[query]),len(self.kappas[query])
          
    #def get_best(self,query):
    #    return ordonne_theta_function_kappa(self.thetas[query],self.kappas[query])
    
    def get_best_index(self, query):
        return order_theta_according_to_kappa_index(self.thetas[query], self.kappas[query])
    
    def get_best_decrease(self, query):
        nb_position = len(self.kappas[query])
        return maximum_K(self.thetas[query],nb_position)
    
    def get_best_index_decrease(self, query):
        nb_position = len(self.kappas[query])
        return maximum_K_index(self.thetas[query], nb_position)
    
    def get_expected_reward(self,propositions,query):
        expec_rew = []
        for pos in range(len(propositions)):
            item = propositions[pos]
            #print(query)
            #print(pos)
            #print(item)
            expec_rew.append(self.kappas[query][pos]*self.thetas[query][item])
        return expec_rew
