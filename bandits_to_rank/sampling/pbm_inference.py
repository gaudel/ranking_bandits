#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Infer parameters of PBM model

Current version: interface to PyClick implementation of EM strategy
"""

import numpy as np
from scipy.linalg import svd

from pyclick.click_models.PBM import PBM
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult




class EM():

    def __init__(self, nb_products, nb_positions):
        self.search_sessions = []
        self.click_model = PBM()
        self.nb_products = nb_products
        self.nb_positions = nb_positions

    def get_nb_products(self):
        return self.nb_products

    def get_nb_positions(self):
        return self.nb_positions

    def add_session(self, propositions, rewards):
        # transform session
        web_results = []
        for index_product, reward in zip(propositions, rewards):
            web_results.append(SearchResult(index_product, reward))
        # add session
        self.search_sessions.append(TaskCentricSearchSession(len(self.search_sessions), 'Reco'))
        self.search_sessions[-1].web_results = web_results

    def learn(self):
        self.click_model.train(self.search_sessions)

    def get_params(self):
        return self.get_thetas(), self.get_kappas()

    def get_thetas(self):
        thetas = np.ones(self.nb_products, dtype=np.float)
        for i in range(self.nb_products):
            thetas[i] = self.click_model.params[PBM.param_names.attr].get('Reco', i).value()
        return thetas

    def get_kappas(self):
        kappas = np.ones(self.nb_positions, dtype=np.float)
        for i in range(self.nb_positions):
            kappas[i] = self.click_model.params[PBM.param_names.exam].get(i).value()
        return kappas


class SVD():
    def __init__(self, nb_products, nb_positions, prior=0.1):
        self.prior = prior
        self.nb_views = np.zeros((nb_products, nb_positions), dtype=np.int) + self.prior
        self.nb_clicks = np.zeros((nb_products, nb_positions), dtype=np.int) + self.prior
        self.positions = np.arange(nb_positions)

    def get_nb_products(self):
        return self.nb_views.shape[0]

    def get_nb_positions(self):
        return len(self.positions)

    def add_session(self, propositions, rewards):
        self.nb_views[propositions, self.positions] += 1
        self.nb_clicks[propositions, self.positions] += rewards

    def learn(self):
        U, d, V = svd(self.nb_clicks/self.nb_views, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=True)
        ind_maxV = np.argmax(abs(V[0,:]))
        maxV = V[0, ind_maxV]

        self.kappas_hat = np.array(V[0,:] / maxV)
        self.thetas_hat = np.array(U[:,0] * maxV * d[0])

    def get_params(self):
        return self.thetas_hat, self.kappas_hat

    def get_thetas(self):
        return self.thetas_hat

    def get_kappas(self):
        return self.kappas_hat



class Oracle():

    def __init__(self, thetas=None, kappas=None):
        self.thetas = np.array(thetas)
        self.kappas = np.array(kappas)

    def get_nb_products(self):
        return self.thetas.shape[0]

    def get_nb_positions(self):
        return self.kappas.shape[0]

    def add_session(self, propositions, rewards):
        pass

    def learn(self):
        pass

    def get_params(self):
        return self.get_thetas(), self.get_kappas()

    def get_thetas(self):
        return self.thetas

    def get_kappas(self):
        return self.kappas


if __name__ == "__main__":
    import doctest

    doctest.testmod()
