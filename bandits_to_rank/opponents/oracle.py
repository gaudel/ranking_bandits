
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


class Oracle:
    """
    Player which plays the best arm
    """
    def __init__(self, best_arm):
        self.best_arm = best_arm
        self.time_reject = 0

    def clean(self):
        pass

    def choose_next_arm(self):
        return self.best_arm, self.time_reject

    def update(self, arm, reward):
        pass
    
    def type(self):
        return 'Oracle'
        