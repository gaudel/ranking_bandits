

from bandits_to_rank.sampling.pbm_inference import *


class GetOracle():
    def __init__(self, kappas):
        self.kappas = kappas

    def __call__(self):
        return Oracle(kappas=self.kappas)


# Wrap the function `get_model()` in a class to enable pickling
class GetEM():
    def __init__(self, nb_arms, nb_positions):
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions

    def __call__(self):
        res = EM(self.nb_arms, self.nb_positions)
        res.nb_views = np.ones((self.nb_arms, self.nb_positions), dtype=np.int) * 0.0001
        return res

class GetSVD():
    def __init__(self, nb_arms, nb_positions, prior_s=0.1,prior_f=0.1):
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.prior = prior_s

    def __call__(self):
        model = SVD(self.nb_arms, self.nb_positions,self.prior)
        model.learn()
        return model



class GetMLE():
    def __init__(self, nb_arms, nb_positions, prior_s=0.5, prior_f=0.5):
        pass

    def __call__(self):
        pass
