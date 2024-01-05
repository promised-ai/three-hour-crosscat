from math import log
from typing import Hashable
import pandas as pd

from .assignment import Assignment
from .model import Model


class Feature:
    data: pd.Series

    def __init__(self, id: int, data: pd.Series, beta_a=0.5, beta_b=0.5):
        self.data = data
        self.components = []
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.id = id

    @property
    def n(self):
        return self.data.shape[0]

    def logm(self, row_ix):
        x = self.data.iloc[row_ix]
        p = self.beta_a / (self.beta_a + self.beta_b)

        if x:
            return log(p)
        else:
            return log(1.0 - p)

    def logpp_z(self, row_ix, z):
        x = self.data.iloc[row_ix]
        return self.components[z].logpp(x)

    def logm_under_assignment(self, asgn: Assignment):
        cpnts = [Model(self.beta_a, self.beta_b) for _ in range(asgn.n_cats)]
        for x, zi in zip(self.data, asgn.zs):
            cpnts[zi].observe(x)

        return sum(cpnt.logm() for cpnt in cpnts)

    def remove_row(self, row_ix: int, z: int):
        x = self.data.iloc[row_ix]
        self.components[z].forget(x)

        if self.components[z].n == 0:
            del self.components[z]

    def insert_datum_into_singleton(self, row_ix):
        x = self.data.iloc[row_ix]
        self.components.append(Model(self.beta_a, self.beta_b))
        self.components[-1].observe(x)

    def insert_datum_into_cat(self, row_ix, z):
        # NOTE: assumes the z is not a singleton
        x = self.data.iloc[row_ix]
        self.components[z].observe(x)

    def set_assignment(self, assignment: Assignment):
        self.components = [Model(self.beta_a, self.beta_b) for 
            _ in range(assignment.n_cats)]
        for x, zi in zip(self.data, assignment.zs):
            self.components[zi].observe(x)


    



