import pickle
import os
import numpy as np
import pandas as pd

items = []
fileOrder = []

with open('./freq_order.pickle','rb') as f:
    order=pickle.load(f)

for file in os.listdir('./top_items'):
    if 'pairs' in file:
        with open('./top_items/{}'.format(file),'rb') as f:
            idx = pickle.load(f)
            fileOrder.append(os.path.basename(f.name))

        if type(idx) != list:
            idx = idx[0].tolist()

        A = []
        if not type(idx[0]) == int:
            for x,y in idx:
                if not order[x] == order[y]:
                    A.append((order[x],order[y]))
        else:
            for x in idx:
                A.append(order[x])

        items.append(A)

data = {}
for idx, i in enumerate(items):
    S = pd.Series(i, name=fileOrder[idx])
    data[S.name] = S

df = pd.DataFrame(data)

#Get +ve and -ve pairs from the betas
with open('./results/coefficients/firstOffs.pickle','rb') as f:
    first=pickle.load(f)

posPairs = []
positives = list(np.linspace(0,0.031))
positives.reverse()

negPairs = []
negatives = list(np.linspace(-0.02,0))
negatives.reverse()

for x in positives:
    idx = list(zip(list(np.where(first > x)[0]), list(np.where(first > x)[1])))
    for i in idx:
        if not i in posPairs:
            posPairs.append(i) 

for x in negatives:
    idx = list(zip(list(np.where(first < x)[0]), list(np.where(first < x)[1])))
    for i in idx:
        if not i in negPairs:
            negPairs.append(i) 

posItems = []
for x in posPairs:
    posItems.append(order[x])

negItems = []
for x in negPairs:
    negItems.append(order[x])
    