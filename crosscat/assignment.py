from typing import List
import numpy as np
from numpy.random import choice


class Assignment:
    alpha: float
    n: int
    n_cats: int
    counts: List[int]
    zs: List[int]

    def __init__(self, n, alpha):
        """Draw from CRP (alpha)"""
        self.n_cats = 1
        self.counts = [1]
        self.zs = [0]
        self.n = n
        self.alpha = alpha

        for i in range(1, n):
            ps = np.array(self.counts + [alpha])
            ps /= np.sum(ps)
            z = choice(len(ps), p=ps)

            if z == self.n_cats:
                # new table
                self.counts.append(1)
                self.n_cats += 1
            else:
                self.counts[z] += 1

            self.zs.append(z)

    def unassign(self, ix: int):
        # NOTE: do we alter arrays in lace? Would not doing that help performance?
        z = self.zs[ix]
        singleton = self.counts[z] == 1

        if singleton:
            del self.counts[z]
            self.n_cats -= 1

            for i in range(self.n):
                if self.zs[i] > z:
                    self.zs[i] -= 1
        else:
            self.counts[z] -= 1

        self.zs[ix] = -1


    def reassign(self, ix, z: int):
        assert self.zs[ix] == -1
        assert z <= self.n_cats

        self.zs[ix] = z
        if z == self.n_cats:
            self.counts.append(1)
            self.n_cats += 1
        else:
            self.counts[z] += 1
