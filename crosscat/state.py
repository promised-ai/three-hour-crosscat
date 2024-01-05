from typing import List
from random import shuffle
import numpy as np
from numpy.random import choice
from scipy.special import logsumexp

from .assignment import Assignment
from .feature import Feature
from .view import View


class State:
    views: List[View]
    assignment: Assignment
    alpha: float

    def __init__(self, features: List[Feature], alpha=1.0):
        n_cols = len(features)
        self.alpha = alpha
        self.assignment = Assignment(n_cols, alpha)
        n_rows = features[0].n

        views = [View([], alpha=alpha) for _ in range(self.assignment.n_cats)]
        for view in views:
            view.set_assignment(Assignment(n_rows, alpha))

        for feature, z in zip(features, self.assignment.zs):
            views[z].insert(feature)

        self.views = views

    @property
    def shape(self):
        n_rows = self.views[0].n
        n_cols = sum(len(view.features) for view in self.views)
        return n_rows, n_cols

    def resassign_rows_gibbs(self):
        for view in self.views:
            view.reassign_rows_gibbs()

    def remove_column_gibbs(self, col_ix):
        view_ix = self.assignment.zs[col_ix]
        self.assignment.unassign(col_ix)
        feature = self.views[view_ix].features.pop(col_ix)

        if self.views[view_ix].n_cols == 0:
            del self.views[view_ix]

        return feature

    def reinsert_column_gibbs(self, feature: Feature, m: int):
        logps = np.log(self.assignment.counts + [self.assignment.alpha/m] * m)

        for view_ix in range(self.assignment.n_cats):
            asgn = self.views[view_ix].assignment
            logps[view_ix] += feature.logm_under_assignment(asgn)

        singletons = []
        for i in range(self.assignment.n_cats, self.assignment.n_cats + m):
            asgn = Assignment(feature.n, self.alpha)
            singletons.append(asgn)
            logps[i] += feature.logm_under_assignment(asgn)

        logps -= logsumexp(logps)
        z = choice(len(logps), p=np.exp(logps))

        if z > len(logps) - m - 1:
            new_view = View([feature], self.alpha)
            singleton_ix = z - len(logps)
            asgn = singletons[singleton_ix]
            new_view.set_assignment(asgn)
            self.views.append(new_view)
            self.assignment.reassign(feature.id, self.assignment.n_cats)
        else:
            self.assignment.reassign(feature.id, z)
            self.views[z].insert(feature)

    def reassign_columns_gibbs(self, m: int=1):
        col_ixs = list(range(self.shape[1]))
        shuffle(col_ixs)

        for col_ix in col_ixs:
            feature = self.remove_column_gibbs(col_ix)
            self.reinsert_column_gibbs(feature, m=m)

    def update(self, n_steps: int):
        for _ in range(n_steps):
            self.reassign_columns_gibbs()
            self.resassign_rows_gibbs()

