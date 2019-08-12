import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

lev = pd.read_csv('./results/association_rules/association_rules_one_indv.csv', sep=',')
leverage = lev[['antecedents','consequents','leverage']]

ant = leverage['antecedents'].tolist()
ant = [i[12:-3] for i in ant]
cons = leverage['consequents'].tolist()
cons = [i[12:-3] for i in cons]
pairs = list(zip(ant, cons))

lev = leverage['leverage'].tolist()
lev = [0 if i<0 else i for i in lev]

data = {k:v for k,v in zip(pairs,lev)}

keys = np.array(list(data.keys()))
vals = np.array(list(data.values()))

unq_keys, key_idx = np.unique(keys, return_inverse=True)
key_idx = key_idx.reshape(-1, 2)
n = len(unq_keys)
adj = np.zeros((n, n))
adj[key_idx[:,0], key_idx[: ,1]] = vals
adj += adj.T

matrix = np.eye(183,183)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(matrix, interpolation='nearest', cmap=cm.Greys_r)
plt.show()