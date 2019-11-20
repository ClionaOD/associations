import random
import numpy as np
import pickle
from statsmodels.tsa.api import VAR
import get_frequent_items as freq

with open('itemsets.pickle', 'rb') as f:
    itemsets = pickle.load(f)

nlags = 5

one_hot = freq.one_hot(itemsets)

model = VAR(one_hot)
results = model.fit(maxlags=nlags)
print(results.summary())