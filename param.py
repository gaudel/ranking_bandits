## Requirements

### Packages
import os
import json
import gzip
import numpy as np

from bandits_to_rank.environment import Environment_PBM, PositionsRanking
from bandits_to_rank.opponents import greedy
from bandits_to_rank.opponents.pbm_pie import PBM_PIE_Greedy_SVD, PBM_PIE_semi_oracle, PBM_PIE_Greedy_MLE
from bandits_to_rank.opponents.pbm_ucb import PBM_UCB_Greedy_SVD, PBM_UCB_semi_oracle, PBM_UCB_Greedy_MLE
from bandits_to_rank.opponents.pbm_ts import PBM_TS_Greedy_SVD, PBM_TS_semi_oracle, PBM_TS_Greedy_MLE
from bandits_to_rank.opponents.bc_mpts import BC_MPTS_Greedy_SVD, BC_MPTS_semi_oracle,BC_MPTS_Greedy_MLE
#from bandits_to_rank.opponents.pmed import PMED   # loaded only before usage to load tensorflow library only when required
from bandits_to_rank.opponents.top_rank import TOP_RANK
from bandits_to_rank.opponents.grab import GRAB
from bandits_to_rank.opponents.f_grab import sGRAB
from bandits_to_rank.opponents.combucb import CombUCB1, KL_CombUCB1
from bandits_to_rank.opponents.pb_mhb import *
from bandits_to_rank.referee import Referee

# set.seed(123)


# Path to bandits-to-rank module
import bandits_to_rank

packagedir = os.path.dirname(bandits_to_rank.__path__[0])


class NdArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def record_zip(filename, dico):
    print(type(dico))
    print('file', filename)
    json_str = json.dumps(dico, cls=NdArrayEncoder)
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile(filename, 'w') as fout:
        fout.write(json_bytes)
    return 'done'


class Parameters():
    """ Parameters used for the experiment

    # Environement
        env
        env_name        str used for name of files
        logs_env_name   (only for merge)

    # Player
        player
        player_name

    # Rules
        rules_name
        referee

    # Sub-experiment
        first_game      (only for play)
        end_game        (only for play)
        input_path      (only for merge)
        output_path
        force           (only for play)
    """

    def __init__(self):
        self.env = Environment_PBM([1], [1], label="fake")
        self.positions_ranking = PositionsRanking.SHUFFLE_EXCEPT_FIRST  # default: shuffle kappas before each game
        self.nb_relevant_positions = None  # default: compute reward at each position
        self.rng = np.random.default_rng()

    #########" PBM_Setting
    def set_positions_ranking(self, positions_ranking):
        self.positions_ranking = positions_ranking

        # tag for file names and logs
        if positions_ranking == PositionsRanking.FIXED:
            raise ValueError('fixed ranking of positions should be set by the player')
        elif positions_ranking == PositionsRanking.DECREASING:
            # TODO: better naming for PBM '__decreasing_kappa'
            # TODO: better naming for CM '__std_order_on_views'
            tag = '__sorted_kappa' if type(self.env) == Environment_PBM else ''
        elif positions_ranking == PositionsRanking.SHUFFLE:
            # TODO: better naming for CM '__random_order_on_views'
            tag = '__shuffled_kappa' if type(self.env) == Environment_PBM else '_order_view_shuffle'
        elif positions_ranking == PositionsRanking.SHUFFLE_EXCEPT_FIRST:
            # TODO: better naming for PBM __shuffled_kappa_except_first
            tag = '' if type(self.env) == Environment_PBM else '__random_order_on_views_except_first'
        elif positions_ranking == PositionsRanking.INCREASING:
            tag = ('__increasing_kappa' if type(self.env) == Environment_PBM else '__reverse_order_on_views')
        elif positions_ranking == PositionsRanking.INCREASING_EXCEPT_FIRST:
            tag = ('__increasing_kappa_except_first' if type(self.env) == Environment_PBM
                   else '__reverse_order_on_views_except_first')
        else:
            raise ValueError(f'unhandled ranking on positions: {positions_ranking}')
        self.env_name += tag
        self.logs_env_name += tag
        self.env.label += tag

    def set_env_KDD_all(self):
        """!!! only to merge logs from several queries of KDD data !!!"""
        self.env_name = f'KDD_all'
        self.logs_env_name = f'KDD_[0-9]*_query'

    def set_env_KDD(self, query):
        # load KDD data
        # todo: to put in bandits_to_rank.data
        with open(packagedir + '/data/param_KDD.txt', 'r') as file:
            dict_theta_query = json.load(file)
        query_name, query_params = list(dict_theta_query.items())[query]

        # set environement
        self.env_name = f'KDD_{query}_query'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(query_params['thetas'], query_params['kappas']
                                   , label='%s (%d for us)' % (query_name, query))

    def set_env_Yandex_all(self):
        """!!! only to merge logs from several queries of Yandex data !!!"""
        self.env_name = f'Yandex_all'
        self.logs_env_name = f'Yandex_[0-9]*_query'

    def set_env_Yandex(self, query):
        # load Yandex data
        # todo: to put in bandits_to_rank.data
        with open(packagedir + '/data/param_Yandex.txt', 'r') as file:
            dict_theta_query = json.load(file)
        query_name, query_params = list(dict_theta_query.items())[query]

        # reduce to 10 products, 5 positions
        thetas = np.sort(query_params['thetas'])[:-11:-1]
        kappas = np.sort(query_params['kappas'])[:-6:-1]

        # set environement
        self.env_name = f'Yandex_{query}_query'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas
                                   , label='%s (%d for us)' % (query_name, query))

    def set_env_Yandex_equi_all(self, K):
        """!!! only to merge logs from several queries of Yandex data !!!"""
        self.env_name = f'Yandex_equi_{K}_K_all'
        self.logs_env_name = f'Yandex_equi_{K}_K__[0-9]*_query'

    def set_env_Yandex_equi(self, query, K):
        # load Yandex data
        with open(packagedir + '/data/param_Yandex.txt', 'r') as file:
            dict_theta_query = json.load(file)
        query_name, query_params = list(dict_theta_query.items())[query]

        # reduce to 10 products, 10 positions
        index_max = 1 + K
        thetas = np.sort(query_params['thetas'])[:-index_max:-1]
        kappas = np.sort(query_params['kappas'])[:-index_max:-1]

        # set environement
        self.env_name = f'Yandex_equi_{K}_K__{query}_query'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label=f'Yandex equi K={K} {query} ({query_name} for Yandex)')

    def set_env_test(self):
        """Purely simulated environment with standard click's probabilities"""
        kappas = [1, 0.6, 0.3]
        thetas = [0.1, 0.5, 0.1, 0.6, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1]
        self.env_name = f'purely_simulated__test'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, test")

    def set_env_std(self):
        """Purely simulated environment with standard click's probabilities"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.3, 0.2, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.01, 0.01]
        self.env_name = 'purely_simulated__std'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, std")

    def set_env_small(self):
        """Purely simulated environment with click's probabilities close to 0"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.15, 0.1, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.env_name = 'purely_simulated__small'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, small")

    def set_env_big(self):
        """Purely simulated environment with click's probabilities close to 1"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75]
        self.env_name = 'purely_simulated__big'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, big")

    def set_env_extra_small(self):
        """Purely simulated environment with click's probabilities close to 0"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
        self.env_name = 'purely_simulated__xsmall'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, extra small")

    def set_env_xx_small(self):
        """Purely simulated environment with click's probabilities close to 0"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
        self.env_name = 'purely_simulated__xxsmall'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, xx small")

    def set_env_simul(self, label):
        if label == "std":
            self.set_env_std()
        elif label == "big":
            self.set_env_big()
        elif label == "small":
            self.set_env_small()
        elif label == "xsmall":
            self.set_env_extra_small()
        elif label == "xxsmall":
            self.set_env_xx_small()
        else:
            raise ValueError("unknown label of environment")

    def set_rules(self, nb_trials, nb_records=1000):
        # Check inputs
        if nb_records > nb_trials:
            nb_records = -1

        self.rules_name = f'games_{nb_trials}_nb_trials_{nb_records}_record_length'
        self.referee = Referee(self.env, nb_trials, all_time_record=False, len_record_short=nb_records)

    def set_player_eGreedy(self, c, update=100, noSVD=False):
        nb_prop, nb_place = self.env.get_setting()
        if noSVD:
            self.player_name = f'Bandit_EGreedy_EM_{c}_c_{update}_update'
            self.player = greedy.greedy_EGreedy_EM(c, nb_prop, nb_place, update)
        else:
            self.player_name = f'Bandit_EGreedy_SVD_{c}_c_{update}_update'
            self.player = greedy.greedy_EGreedy(c, nb_prop, nb_place, update)

    def set_player_PBM_TS(self, type="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = 'Bandit_PBM-TS_oracle'
            self.player = PBM_TS_semi_oracle(nb_prop, nb_place, discount_factor=self.env.kappas, count_update=1)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_PBM_TS_greedy_MLE'
            self.player = PBM_TS_Greedy_MLE(nb_prop, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_PBM_TS_greedy_SVD'
            self.player = PBM_TS_Greedy_SVD(nb_prop, nb_place, count_update=1)
        else:
            self.player_name = 'Bandit_PBM-TS_greedy_SVD'
            self.player = PBM_TS_Greedy_SVD(nb_prop, nb_place, count_update=1)

    def set_player_PBM_PIE(self, epsilon, T, type ="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = f'Bandit_PBM-PIE_oracle_{epsilon}_epsilon'
            self.player = PBM_PIE_semi_oracle(nb_prop, epsilon, T, nb_place, discount_factor=self.env.kappas, count_update=1)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_PBM_PIE_greedy_MLE'
            self.player = PBM_PIE_Greedy_MLE(nb_prop, epsilon, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_PBM_PIE_greedy_SVD'
            self.player = PBM_PIE_Greedy_SVD(nb_prop, epsilon, nb_place, count_update=1)
        else:
            self.player_name = f'Bandit_PBM-PIE_greedy_SVD_{epsilon}_epsilon'
            self.player = PBM_PIE_Greedy_SVD(nb_prop, epsilon, T, nb_place, count_update=1)

    def set_player_PBM_UCB(self, epsilon, type ="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = f'Bandit_PBM_UCB_oracle_{epsilon}_epsilon'
            self.player = PBM_UCB_semi_oracle(nb_prop, epsilon, nb_place, discount_factor=self.env.kappas, count_update=1)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_PBM_UCB_greedy_MLE'
            self.player = PBM_UCB_Greedy_MLE(nb_prop, epsilon, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_PBM_UCB_greedy_SVD'
            self.player = PBM_UCB_Greedy_SVD(nb_prop, epsilon, nb_place, count_update=1)
        else:
            self.player_name = f'Bandit_PBM_UCB_greedy_SVD_{epsilon}_epsilon'
            self.player = PBM_UCB_Greedy_SVD(nb_prop, epsilon, nb_place, count_update=1)


    def set_player_BC_MPTS(self, type ="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = 'Bandit_BC-MPTS_oracle'
            self.player = BC_MPTS_semi_oracle(nb_prop, nb_place, self.env.kappas)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_BC-MPTS_greedy_MLE'
            self.player = BC_MPTS_Greedy_MLE(nb_prop, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_BC-MPTS_greedy_SVD'
            self.player = BC_MPTS_Greedy_SVD(nb_prop, nb_place, count_update=1)
        else:
            self.player_name = 'Bandit_BC-MPTS_greedy_SVD'
            self.player = BC_MPTS_Greedy_SVD(nb_prop, nb_place, count_update=1)

    def set_player_PMED(self, alpha, gap_MLE, gap_q, run=True):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_PMED_{alpha}_alpha_{gap_MLE}_gap_MLE_{gap_q}_gap_q'

        if run:
            from bandits_to_rank.opponents.pmed import PMED
            self.player = PMED(nb_prop, nb_place, alpha, gap_MLE, gap_q)

    def set_player_CombUCB1(self, exploration_factor=2.):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_CombUCB1_{exploration_factor}_exploration'
        self.player = CombUCB1(nb_arms=nb_prop, nb_positions=nb_place, exploration_factor=exploration_factor)

    def set_player_KL_COMB(self, horizon):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_KL-COMB_{horizon}_horizon'
        self.player = KL_CombUCB1(nb_arms=nb_prop, nb_positions=nb_place, horizon=horizon)

    def set_player_PB_MHB(self, nb_steps, random_start=False):
        nb_prop, nb_place = self.env.get_setting()
        if random_start:
            self.player_name = f'Bandit_PB-MHB_random_start_{nb_steps}_step_{self.proposal_name}_proposal'
            self.player = PB_MHB(nb_prop, nb_place, proposal_method=self.proposal, step=nb_steps,
                                 part_followed=False)
        else:
            self.player_name = f'Bandit_PB-MHB_warm-up_start_{nb_steps}_step_{self.proposal_name}_proposal'
            self.player = PB_MHB(nb_prop, nb_place, proposal_method=self.proposal, step=nb_steps,
                                 part_followed=True)

    def set_player_TopRank(self, T, horizon_time_known=True, doubling_trick=False, oracle=False):
        nb_prop, nb_place = self.env.get_setting()
        if oracle:
            self.player_name = f'Bandit_TopRank_oracle_{T}_delta_{"TimeHorizonKnown" if horizon_time_known else ""}_{"doubling_trick" if doubling_trick else ""}'
            self.player = TOP_RANK(nb_arms=nb_prop,
                                   T=T, horizon_time_known=horizon_time_known,doubling_trick_active=doubling_trick,
                                   discount_factor=self.env.kappas)
            self.positions_ranking = PositionsRanking.FIXED
        else:
            self.player_name = f'Bandit_TopRank_{T}_delta_{"TimeHorizonKnown" if horizon_time_known else ""}_{"doubling_trick" if doubling_trick else ""}'
            self.player = TOP_RANK(nb_arms=nb_prop,
                                   T=T, horizon_time_known=horizon_time_known, doubling_trick_active=doubling_trick,
                                   discount_factor=np.arange(nb_place - 1, -1, -1))
        """
            self.player_name = f'Bandit_TopRank_greedy_{T}_delta_{"TimeHorizonKnown" if horizon_time_known else ""}_{"doubling_trick" if doubling_trick else ""}'
            self.player = TOP_RANK(nb_arms=nb_prop,
                                   T=T, horizon_time_known=horizon_time_known,doubling_trick_active=doubling_trick,
                                   nb_positions=nb_place, lag=1)
        """


    def set_player_SGRAB(self, T, gamma, forced_initiation):
        nb_prop, nb_place = self.env.get_setting()
        if gamma == 0:
            gamma_use = nb_prop * nb_place
        else:
            gamma_use = gamma
        self.player_name = f'Bandit_SGRAB_{T}_T_{gamma_use}_gamma{"_forced" if forced_initiation else ""}'
        self.player = sGRAB(nb_arms=nb_prop, nb_positions=nb_place, T=T)

    def set_player_GRAB(self, T, gamma, forced_initiation):
        nb_prop, nb_place = self.env.get_setting()
        if gamma == 0:
            gamma_use = nb_prop - nb_place
        else:
            gamma_use = gamma
        self.player_name = f'Bandit_GRAB_{T}_T_{gamma_use}_gamma{"_forced" if forced_initiation else ""}'
        self.player = GRAB(nb_arms=nb_prop, nb_positions=nb_place, T=T, gamma=gamma_use,
                                forced_initiation=forced_initiation)

    def set_proposal_TGRW(self, c, vari_sigma=True):
        self.proposal_name = f'TGRW_{c}_c{"_vari_sigma" if vari_sigma else ""}'
        self.proposal = propos_trunk_GRW(c, vari_sigma)

    def set_proposal_LGRW(self, c, vari_sigma=True):
        self.proposal_name = f'LGRW_{c}_c{"_vari_sigma" if vari_sigma else ""}'
        self.proposal = propos_logit_RW(c, vari_sigma)

    def set_proposal_RR(self, c, str_proposal_possible, vari_sigma=True):
        list_proposal_possible = list(str_proposal_possible.split("-"))
        self.proposal_name = f'RR_{c}_c_{len(list_proposal_possible)}_proposals'
        self.proposal = propos_Round_Robin(c, vari_sigma, list_proposal_possible)

    def set_proposal_MaxPos(self):
        self.proposal_name = f'MaxPos'
        self.proposal = propos_max_position()

    def set_proposal_PseudoView(self):
        self.proposal_name = f'PseudoView'
        self.proposal = propos_pseudo_view()

    def set_exp(self, first_game=-1, nb_games=-1, nb_checkpoints=10, input_path=None, output_path=None, force=True):
        self.first_game = first_game
        self.end_game = first_game + nb_games
        self.nb_checkpoints = nb_checkpoints
        self.input_path = input_path if input_path is not None else output_path
        self.output_path = output_path if output_path is not None else input_path
        self.force = force


