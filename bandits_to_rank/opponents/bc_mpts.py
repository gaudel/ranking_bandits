
#### Bandits

## Packages

from bandits_to_rank.sampling.pbm_inference import *
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index
from bandits_to_rank.tools.get_inference_model import GetOracle, GetSVD, GetMLE


from random import sample
from random import random


from numpy.random import beta 



class BC_MPTS:
    """
    Source : "Optimal Regret Analysis of TS in Stochastic MAB Problem with multiple Play"_Komiyama,Honda,Nakagawa
    Approximate random sampling given posterior probability to be optimal,.

    Requires a way to define kappa.
    """

    def __init__(self, nb_arms, nb_position, get_model_kappa, prior_s=0.5, prior_f=0.5, count_update=1):
        self.get_model_kappa = get_model_kappa
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.positions = np.arange(self.nb_position)
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.count_update = count_update
        self.clean()

    def clean(self):
        self.nb_success = np.zeros((self.nb_arms, self.nb_position), dtype=np.uint) + self.prior_s
        self.nb_prints = np.zeros((self.nb_arms, self.nb_position), dtype=np.uint) + self.prior_s + self.prior_f
        self.model_kappa = self.get_model_kappa()
        _, self.discount_factor = self.model_kappa.get_params()
        self.nb_trials = 0
        self.time_reject = 0

    def choose_next_arm(self):
        pseudo_fail = self.nb_prints * self.discount_factor - self.nb_success
        pseudo_fail[pseudo_fail < 0] = 0
        thetas = beta(np.sum(self.nb_success, axis=1) + 1, np.sum(pseudo_fail, axis=1) + 1)
        return order_theta_according_to_kappa_index(thetas, self.discount_factor), self.time_reject

    def update(self, propositions, rewards):
        # update model of kappa
        self.nb_trials += 1
        self.model_kappa.add_session(propositions, rewards)
        if self.nb_trials <= 100 or self.nb_trials % self.count_update == 0:
            self.model_kappa.learn()
            _, self.discount_factor = self.model_kappa.get_params()

        # update BC_MPTS model
        self.nb_prints[propositions, self.positions] += 1
        self.nb_success[propositions, self.positions] += rewards

    def get_param_estimation(self):
        alpha = np.sum(self.nb_success, axis=1) + 1
        pseudo_fail = self.nb_prints * self.discount_factor - self.nb_success
        pseudo_fail[pseudo_fail < 0] = 0
        beta = np.sum(pseudo_fail, axis=1) + 1
        beta[beta < 0] = 0
        theta_estime = (alpha-1)/(alpha+beta -2)
        theta_estime[theta_estime<0]=0
        return theta_estime,self.discount_factor


    def type(self):
        return 'BC_MPTS'


def BC_MPTS_semi_oracle(nb_arms, nb_position, discount_factor, prior_s=0.5, prior_f=0.5, count_update=1):
    """
    BC-MPTS, where kappa is known.
    """
    return BC_MPTS(nb_arms, nb_position, GetOracle(discount_factor), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def BC_MPTS_Greedy_SVD(nb_arms, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    BC-MPTS, where kappa is inferred assuming on rank-1 model (equivalent to PBM), with parameters inferred through SVD of empirical click-probabilities.
    """
    return BC_MPTS(nb_arms, nb_position, GetSVD(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)



def BC_MPTS_Greedy_MLE(nb_arms, nb_position, count_update=1, prior_s=0.5, prior_f=0.5):
    """
    BC-MPTS, where kappa is inferred through MLE of empirical click-probabilities.
    """
    return BC_MPTS(nb_arms, nb_position, GetMLE(nb_arms, nb_position, prior_s, prior_f), prior_s=prior_s, prior_f=prior_f, count_update=count_update)

