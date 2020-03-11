import pickle
import os
import numpy as np
import pandas as pd

items = []
fileOrder = []

with open('./freq_order.pickle','rb') as f:
    order=pickle.load(f)

for file in os.listdir('./top_items'):

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