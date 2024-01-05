from numpy.random import rand
import numpy as np
import pandas as pd

from crosscat.view import View
from crosscat.feature import Feature

n_rows = 33
n_cols = 4
xs = np.vstack([
    np.zeros((n_rows//3, n_cols), dtype=int),
    np.hstack([
        np.zeros((n_rows//3, n_cols//2), dtype=int),
        np.ones((n_rows//3, n_cols//2), dtype=int),
    ]),
    np.ones((n_rows//3, n_cols), dtype=int),
], dtype=int)

# xs = np.bitwise_xor(xs, np.array(rand(n_rows, n_cols) < 0.1, dtype=int))


data = pd.DataFrame(xs)
features = [Feature(data[col]) for col in data]
view = View(features)

print(view.assignment.zs)
for i in range(50):
    view.reassign_rows_gibbs()
    print(view.assignment.zs)
