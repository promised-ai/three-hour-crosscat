from math import log
from scipy.special import betaln


class Model:
    # bernoulli sufficient statistics
    n: int
    n_true: int

    # prior parameters
    beta_a: float
    beta_b: float

    def __init__(self, beta_a, beta_b):
        self.beta_a = beta_a
        self.beta_b = beta_b

        self.n = 0
        self.n_true = 0

    def observe(self, x):
        self.n += 1
        if x:
            self.n_true += 1

    def forget(self, x):
        self.n -= 1
        if x:
            self.n_true -= 1

    def logpp(self, x):
        p = (self.n_true + self.beta_a) / (self.n + self.beta_a + self.beta_b)

        if x:
            return log(p)
        else:
            return log(1.0 - p)

    def logm(self):
        a_prime = self.n_true + self.beta_a
        b_prime = (self.n - self.n_true) + self.beta_b

        return betaln(a_prime, b_prime) - betaln(self.beta_a, self.beta_b)
