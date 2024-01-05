from typing import Hashable, List, Dict
from random import shuffle
import numpy as np
from numpy.random import choice
from scipy.special import logsumexp

from .assignment import Assignment
from .feature import Feature


class View:
    features: Dict[int, Feature]
    alpha: float
    assignment: Assignment

    def __init__(self, features: List[Feature], alpha=1.0):
        self.features = {f.id: f for f in features}

        if len(self.features) > 0:
            self.assignment = Assignment(features[0].data.shape[0], alpha)
            for feature in self.features.values():
                feature.set_assignment(self.assignment)

    @property
    def n(self):
        return self.assignment.n

    @property
    def n_cols(self):
        return len(self.features)

    def set_assignment(self, assignment: Assignment):
        self.assignment = assignment
        for feature in self.features.values():
            feature.set_assignment(assignment)

    def insert(self, feature: Feature):
        feature.set_assignment(self.assignment)
        self.features[feature.id] = feature

    def remove_row_gibbs(self, row_ix: int):
        z = self.assignment.zs[row_ix]
        self.assignment.unassign(row_ix)

        for feature in self.features.values():
            feature.remove_row(row_ix, z)

    def reinsert_row_gibbs(self, row_ix: int):
        # compute p(z_i = k | zs) * p(x_i | y_k) 
        logps = np.log(self.assignment.counts + [self.assignment.alpha])

        for k in range(self.assignment.n_cats):
            for feature in self.features.values():
                assert self.assignment.n_cats == len(feature.components)
                # compute log p(x_i|y_k)
                # - x_i is the datum at row i (row_ix)
                # - y_z is the data assigned to category k
                logps[k] += feature.logpp_z(row_ix, k)

        # probability of singleton
        for feature in self.features.values():
            logps[-1] += feature.logm(row_ix)

        logps -= logsumexp(logps)
        z = choice(len(logps), p=np.exp(logps))

        self.assignment.reassign(row_ix, z)

        if z == len(logps) - 1:
            for feature in self.features.values():
                feature.insert_datum_into_singleton(row_ix)
        else:
            for feature in self.features.values():
                feature.insert_datum_into_cat(row_ix, z)

            
    def reassign_rows_gibbs(self):
        row_ixs = list(range(self.n))
        shuffle(row_ixs)

        for row_ix in row_ixs:
            self.remove_row_gibbs(row_ix)
            self.reinsert_row_gibbs(row_ix)
