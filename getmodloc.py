from sklearn.linear_model import Ridge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE

import numpy as np
import pickle

with open('LCH_allcoefs.pickle', 'rb') as f:
    aggFive = pickle.load(f)

with open('LCH_allcoefs_agg150.pdf', 'rb') as f:
    aggOneFifty = pickle.load(f)

with open('LCH_allcoefs_agg500.pdf', 'rb') as f:
    aggFiveHund = pickle.load(f)


import sys
sys.path.insert(1, '/home/CUSACKLAB/clionaodoherty/associations/statsmodels')

import statsmodels

print(statsmodels.__file__)


