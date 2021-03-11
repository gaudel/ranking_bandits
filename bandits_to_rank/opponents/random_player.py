
#### Bandits

## Packages


import numpy as np
import random as rd


from random import sample
from random import random


from numpy.random import beta 
from random import uniform
from copy import deepcopy


from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import partial



## Algo


class Random:
    
    """
    Player  which plays random arms
    """
    def __init__(self, nb_arm, nb_choice):
        self.nb_arm=nb_arm
        self.nb_choice=nb_choice
        self.time_reject = 0
    def choose_next_arm(self):
        return rd.sample(range(self.nb_arm), self.nb_choice), self.time_reject

    def update(self, arm, reward):
        pass
    
    def type(self):
        return 'Random'
    
  
        