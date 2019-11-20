import random
import numpy as np
import pickle
from statsmodels.tsa.api import VAR
from mlxtend.preprocessing import TransactionEncoder

with open('itemsets.pickle', 'rb') as f:
    itemsets = pickle.load(f)

nlags = 3

te = TransactionEncoder()
one_hot = te.fit(itemsets).transform(itemsets, sparse=False)
one_hot = one_hot.astype(int)

model = VAR(one_hot)
results = model.fit(maxlags=nlags)
print(results.summary())