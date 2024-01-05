from random import shuffle
from numpy.random import rand
import numpy as np
import pandas as pd

from crosscat.state import State
from crosscat.feature import Feature

n_rows = 33

n_cols = 10
xs = np.vstack([
    np.zeros((n_rows//3, n_cols), dtype=int),
    np.hstack([
        np.zeros((n_rows//3, n_cols//2), dtype=int),
        np.ones((n_rows//3, n_cols//2), dtype=int),
    ]),
    np.ones((n_rows//3, n_cols), dtype=int),
], dtype=int)

# xs = np.bitwise_xor(xs, np.array(rand(n_rows, n_cols) < 0.1, dtype=int))

n_cols = 10
ys = np.vstack([
    np.zeros((n_rows//3, n_cols), dtype=int),
    np.hstack([
        np.zeros((n_rows//3, n_cols//2), dtype=int),
        np.ones((n_rows//3, n_cols//2), dtype=int),
    ]),
    np.ones((n_rows//3, n_cols), dtype=int),
], dtype=int)

ixs = list(range(n_rows))
shuffle(ixs)
ys = ys[ixs, :]

data = np.hstack((xs, ys))

data = pd.DataFrame(data)

# data = pd.read_csv('/Users/bax/Documents/promised-ai/lace/lace/resources/datasets/animals/data.csv', index_col=0)
features = [Feature(id, data[col]) for id, col in enumerate(data)]
state = State(features)

print(state.assignment.zs)
for i in range(50):
    state.update(1)
    print(state.assignment.zs)
