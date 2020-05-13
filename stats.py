import os
import pickle
import collections
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd
from scipy import stats
from skbio.stats.distance import mantel

with open('./results/imagenet_categs.pickle','rb') as f:
    coefs = pickle.load(f)

lags = ['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9']

clusters = {}
with open('/home/clionaodoherty/Desktop/CMC/global_categs.pickle', 'rb') as f:
    categories = pickle.load(f)
for k, lst in categories.items():
    for label in lst:
        clusters[label] = k
clusters = {k:v for k,v in clusters.items() if k in coefs['lag_0'].index}
cluster_df = pd.DataFrame.from_dict(clusters, orient='index')

stats_df = pd.DataFrame(index=lags, columns=['corr'])
sig_df = pd.DataFrame(index=lags, columns=['sig'])

model_rdm = pd.DataFrame(data=np.ones((25,25))*-1,index=coefs['lag_0'].index, columns=coefs['lag_0'].index)
for k1, v1 in cluster_df.iterrows():
    for k2, v2 in cluster_df.iterrows():
        if v1[0] == v2[0]:
            model_rdm.loc[k1][k2] = 1
            model_rdm.loc[k2][k1] = 1
np.fill_diagonal(model_rdm.values, 0)

for k, df in coefs.items():
    df_sym = (df + df.T) /2
    np.fill_diagonal(df_sym.values, 0)
    corr, pval, n = mantel(df_sym.values, model_rdm.values, method='kendalltau')
    stats_df.loc[k]['corr'] = corr
    if pval < (0.05 / 10):
        sig_df.loc[k]['sig'] = 1
    else:
        sig_df.loc[k]['sig'] = 0

fig, (ax1,leg) = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [1,.3]})
stats_df.plot.line(ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
leg.legend(handles, labels)
leg.axis('off')

sigs=list(np.where(sig_df==1)[0])
for x in sigs:
    anot = (x , stats_df.iloc[x])
    ax1.annotate('*', anot)
plt.show()
