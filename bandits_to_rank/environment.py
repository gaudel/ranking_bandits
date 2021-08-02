### Environment 

## Packages

from random import random
import random as rd
import numpy as np
from enum import Enum, auto

# from bandits import maximum_K_index, maximum_K
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index, maximum_K_index, maximum_K


### Helping Fonctions

## Environment


class PositionsRanking(Enum):
    FIXED = auto()
    DECREASING = auto()
    SHUFFLE = auto()
    SHUFFLE_EXCEPT_FIRST = auto()
    INCREASING = auto()
    INCREASING_EXCEPT_FIRST = auto()


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

    def shuffle(self, positions_ranking=PositionsRanking.FIXED):
        """Shuffle items and positions

        >>> from GRAB.bandits_to_rank.environment import Environment_PBM, PositionsRanking
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

        >>> env.rng = np.random.default_rng(1)
        >>> env.shuffle(PositionsRanking.SHUFFLE_EXCEPT_FIRST)
        >>> env.thetas
        array([0.2, 0.9, 0.8, 0.3, 0.5, 0.1, 0.4])
        >>> env.get_best_index_decrease()
        array([1, 2, 4, 6, 3])
        >>> env.kappas
        array([1. , 0.4, 0.3, 0.5, 0.7])
        >>> env.get_best_index()
        array([1, 6, 3, 4, 2])

        >>> env.shuffle(PositionsRanking.DECREASING)
        >>> env.thetas
        array([0.9, 0.2, 0.4, 0.1, 0.5, 0.8, 0.3])
        >>> env.kappas
        array([1. , 0.7, 0.5, 0.4, 0.3])

        >>> env.shuffle(PositionsRanking.SHUFFLE)
        >>> env.thetas
        array([0.2, 0.1, 0.9, 0.3, 0.5, 0.8, 0.4])
        >>> env.kappas
        array([0.3, 0.4, 0.7, 1. , 0.5])

        >>> env.shuffle(PositionsRanking.INCREASING)
        >>> env.thetas
        array([0.2, 0.9, 0.8, 0.1, 0.3, 0.5, 0.4])
        >>> env.kappas
        array([0.3, 0.4, 0.5, 0.7, 1. ])

        >>> env.shuffle(PositionsRanking.INCREASING_EXCEPT_FIRST)
        >>> env.thetas
        array([0.2, 0.1, 0.9, 0.8, 0.4, 0.3, 0.5])
        >>> env.kappas
        array([1. , 0.3, 0.4, 0.5, 0.7])
        """
        # thetas
        self.rng.shuffle(self.thetas)

        # kappas
        if positions_ranking is PositionsRanking.FIXED:
            pass
        elif positions_ranking is PositionsRanking.DECREASING:
            self.kappas.sort()
            self.kappas = self.kappas[::-1]
        elif positions_ranking is PositionsRanking.SHUFFLE:
            self.rng.shuffle(self.kappas)
        elif positions_ranking is PositionsRanking.SHUFFLE_EXCEPT_FIRST:
            self.kappas.sort()
            self.kappas = self.kappas[::-1]
            self.rng.shuffle(self.kappas[1:])
        elif positions_ranking is PositionsRanking.INCREASING:
            self.kappas.sort()
        elif positions_ranking is PositionsRanking.INCREASING_EXCEPT_FIRST:
            self.kappas.sort()
            self.kappas = self.kappas[::-1]
            self.kappas[1:].sort()
        else:
            raise ValueError(f'unhandled ranking on positions: {positions_ranking}')

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

    def shuffle(self, positions_ranking=PositionsRanking.FIXED):
        """Shuffle items and positions

        >>> from GRAB.bandits_to_rank.environment import Environment_multirequest_PBM, PositionsRanking
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
        raise NotImplementedError()

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


class Environment_Cascade:
    """
    Describe the comportement of a user in front of a list of item
    Returns a list of rewards : r_k = 1  with probability  tehta_k  and 0 otherwise

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=3)

    >>> thetas = [0.1, 0.5, 0.7, 0.3]
    >>> env = Environment_Cascade(thetas, np.arange(3))
    >>> env_dec = Environment_Cascade(thetas, np.arange(2,-1,-1))

    >>> env.position_index_to_view_index
    array([0, 1, 2])
    >>> env_dec.position_index_to_view_index
    array([2, 1, 0])
    >>> Environment_Cascade(thetas, np.array([0, 3, 1, 2])).position_index_to_view_index
    array([0, 2, 3, 1])

    >>> arm = np.array([0, 1, 2])
    >>> round(env_dec.get_expected_reward(arm).sum(),3), env.get_expected_reward(arm)
    (0.865, array([0.1  , 0.45 , 0.315]))
    >>> arm = np.array([2, 1, 0])
    >>> round(env_dec.get_expected_reward(arm).sum(),3), env_dec.get_expected_reward(arm)
    (0.865, array([0.315, 0.45 , 0.1  ]))

    >>> arm = np.array([2, 3, 1])
    >>> round(env_dec.get_expected_reward(arm).sum(),3), env.get_expected_reward(arm)
    (0.895, array([0.7  , 0.09 , 0.105]))
    >>> arm = np.array([1, 3, 2])
    >>> round(env_dec.get_expected_reward(arm).sum(),3), env_dec.get_expected_reward(arm)
    (0.895, array([0.105, 0.09 , 0.7  ]))
    >>> arm = np.array([1, 2, 3])
    >>> round(env_dec.get_expected_reward(arm).sum(),3), env.get_expected_reward(arm)
    (0.895, array([0.5  , 0.35 , 0.045]))
    """

    def __init__(self, thetas, order_view, label=None):
        self.thetas = np.array(thetas)
        self.nb_position = len(order_view)
        self.label = label
        self.rng = np.random.default_rng()
        self.set_order_view(order_view)

    def set_order_view(self, order_view):
        self.view_index_to_position_index = order_view
        self.position_index_to_view_index = np.argsort(order_view)

    def shuffle(self, positions_ranking=PositionsRanking.FIXED):
        """Shuffle items and positions

        >>> from GRAB.bandits_to_rank.environment import Environment_Cascade, PositionsRanking
        >>> import random
        >>> import numpy as np
        >>> np.set_printoptions(precision=2)

        >>> thetas = [0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
        >>> env = Environment_Cascade(thetas, np.arange(5))

        >>> env.get_best_index_decrease()
        array([0, 1, 2, 3, 4])
        >>> env.get_best_index()
        array([0, 1, 2, 3, 4])

        >>> env.rng = np.random.default_rng(1)
        >>> env.shuffle(PositionsRanking.SHUFFLE_EXCEPT_FIRST)
        >>> env.thetas
        array([0.2, 0.9, 0.8, 0.3, 0.5, 0.1, 0.4])
        >>> env.get_best_index_decrease()
        array([1, 2, 4, 6, 3])
        >>> env.view_index_to_position_index
        array([0, 3, 4, 2, 1])
        >>> env.get_best_index()
        array([1, 6, 3, 4, 2])

        >>> env.shuffle(PositionsRanking.DECREASING)
        >>> env.thetas
        array([0.9, 0.2, 0.4, 0.1, 0.5, 0.8, 0.3])
        >>> env.view_index_to_position_index
        array([0, 1, 2, 3, 4])

        >>> env.shuffle(PositionsRanking.SHUFFLE)
        >>> env.thetas
        array([0.2, 0.1, 0.9, 0.3, 0.5, 0.8, 0.4])
        >>> env.view_index_to_position_index
        array([4, 3, 1, 0, 2])

        >>> env.shuffle(PositionsRanking.INCREASING)
        >>> env.thetas
        array([0.2, 0.9, 0.8, 0.1, 0.3, 0.5, 0.4])
        >>> env.view_index_to_position_index
        array([4, 3, 2, 1, 0])

        >>> env.shuffle(PositionsRanking.INCREASING_EXCEPT_FIRST)
        >>> env.thetas
        array([0.2, 0.1, 0.9, 0.8, 0.4, 0.3, 0.5])
        >>> env.view_index_to_position_index
        array([0, 4, 3, 2, 1])
        """
        # thetas
        self.rng.shuffle(self.thetas)

        # positions
        if positions_ranking is PositionsRanking.FIXED:
            pass
        elif positions_ranking is PositionsRanking.DECREASING:
            self.view_index_to_position_index = np.arange(self.nb_position)
        elif positions_ranking is PositionsRanking.SHUFFLE:
            self.rng.shuffle(self.view_index_to_position_index)
        elif positions_ranking is PositionsRanking.SHUFFLE_EXCEPT_FIRST:
            self.view_index_to_position_index.sort()
            self.rng.shuffle(self.view_index_to_position_index[1:])
        elif positions_ranking is PositionsRanking.INCREASING:
            self.view_index_to_position_index = np.arange(self.nb_position - 1, -1, -1)
        elif positions_ranking is PositionsRanking.INCREASING_EXCEPT_FIRST:
            self.view_index_to_position_index = np.arange(self.nb_position, 0, -1)
            self.view_index_to_position_index[0] = 0
        else:
            raise ValueError(f'unhandled ranking on positions: {positions_ranking}')
        self.position_index_to_view_index = np.argsort(self.view_index_to_position_index)

    def get_reward(self, propositions):
        """
        get vector of probability to look at a each position, given the item in each positions
        $P(o_i) = \prod_{l=0}^{i-1} (1-\theta_l)$, for each $i$ in $\{0, ..., L-1\}$

        Parameters
        ----------
        propositions

        Returns
        -------

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(precision=2)

        >>> thetas = [0.1, 0.5, 0.6, 0.3]
        >>> env = Environment_Cascade(thetas, np.arange(3))
        >>> env_dec = Environment_Cascade(thetas, np.arange(2,-1,-1))

        >>> propositions = np.array([0, 1, 2])
        >>> env.get_expected_reward(propositions)
        array([0.1 , 0.45, 0.27])
        >>> n = 100000
        >>> stats = np.zeros(len(propositions))
        >>> for _ in range(n): stats += env.get_reward(propositions)
        >>> stats / n
        array([0.1 , 0.45, 0.27])

        >>> propositions = np.array([2, 1, 0])
        >>> env_dec.get_expected_reward(propositions)
        array([0.27, 0.45, 0.1 ])
        >>> stats = np.zeros(len(propositions))
        >>> for _ in range(n): stats += env_dec.get_reward(propositions)
        >>> stats / n
        array([0.27, 0.45, 0.1 ])
        """
        click_probabilities = np.concatenate((self.get_expected_reward(propositions), np.zeros(1)))
        return self.rng.multinomial(1, click_probabilities)[:-1]

    def _thetas(self):
        return self.thetas

    def _kappas(self):
        return np.array([0. for i in range(self.nb_position)])

    def get_setting(self):
        return len(self.thetas), self.nb_position

    def get_best_index(self):
        return np.array(self.thetas).argsort()[::-1][self.view_index_to_position_index]

    def get_best_decrease(self):
        theta_ordered = np.sort(np.array(self.thetas))
        return theta_ordered[::-1]

    def get_best_index_decrease(self):
        return maximum_K_index(self.thetas, self.nb_position)

    def get_expected_reward(self, propositions):
        """
        get vector of probability to look at a each position, given the item in each positions
        $P(o_i) = \prod_{l=0}^{i-1} (1-\theta_l)$, for each $i$ in $\{0, ..., L-1\}$

        Parameters
        ----------
        propositions

        Returns
        -------

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(precision=3)

        >>> thetas = [0.1, 0.5, 0.7, 0.3]
        >>> env = Environment_Cascade(thetas, np.arange(3))
        >>> env_dec = Environment_Cascade(thetas, np.arange(2,-1,-1))

        >>> arm = np.array([0, 1, 2])
        >>> env.get_expected_reward(arm)
        array([0.1  , 0.45 , 0.315])
        >>> arm = np.array([2, 1, 0])
        >>> env_dec.get_expected_reward(arm)
        array([0.315, 0.45 , 0.1  ])
        """
        return self.observation_probabilities(propositions) * self.thetas[propositions]

    def observation_probabilities(self, propositions):
        """
        get vector of probability to look at a each position, given the item in each positions
        $P(o_i) = \prod_{l=0}^{i-1} (1-\theta_l)$, for each $i$ in $\{0, ..., L-1\}$

        Parameters
        ----------
        propositions

        Returns
        -------

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(precision=3)

        >>> thetas = [0.1, 0.5, 0.7, 0.3]
        >>> env = Environment_Cascade(thetas, np.arange(3))
        >>> env_dec = Environment_Cascade(thetas, np.arange(2,-1,-1))

        >>> arm = np.array([0, 1, 2])
        >>> env.observation_probabilities(arm)
        array([1.  , 0.9 , 0.45])
        >>> arm = np.array([2, 1, 0])
        >>> env_dec.observation_probabilities(arm)
        array([0.45, 0.9 , 1.  ])
        """
        res = np.ones(self.nb_position)
        np.cumprod((1 - self.thetas[propositions])[self.view_index_to_position_index[:-1]], out=res[1:])
        return res[self.position_index_to_view_index]

    def get_params(self):
        return {"label": self.label, "thetas": self.thetas, "order_view": self.view_index_to_position_index}

