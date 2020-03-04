import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy import stats

def hierarchical_clustering(matrix, label_list, outpath=None):
    fig,ax = plt.subplots(figsize=(10,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), ax=ax, labels=label_list)
    ax.tick_params(axis='x', labelsize=4)
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order

loadPath = './results/coefficients'
orderPath = './freq_order.pickle'
savePath = './results/figs/matrices/updated'

with open('{}/all-betas.pickle'.format(loadPath), 'rb') as f:
    all_betas = pickle.load(f)

with open(orderPath,'rb') as f:
    order = pickle.load(f)

mins = [3,15,40,50,75,100,120] #mins chosen to analyse
matrix_lags = [1,5,13,17,24,33,-1] #indices of the lags chosen from R2 graph from temporal_regression.py

mean_coefs = np.mean(all_betas, axis=3)

#get hierarchical clustering order of the first regression 
clusterOrder = hierarchical_clustering(mean_coefs[0], order)

for idx, lag in enumerate(matrix_lags):
    fig,ax = plt.subplots(figsize=[20,15])
    sns.heatmap(mean_coefs[:,:,lag],ax=ax, cmap='YlGnBu', xticklabels=clusterOrder, yticklabels=clusterOrder)
    ax.axes.set_title('Mean coefficients {} min lag'.format(mins[idx]), fontsize=45)
    ax.tick_params(labelsize=7)
    plt.savefig('{}/meanCoefs_{}mins.pdf'.format(savePath, mins[idx]))

"""

lags = [1,4500,13500,22500,31500,40500]
nlags = len(lags)

chosenOrder = lchOrder

if chosenOrder == lchOrder:
    tag = 'LCH'
elif chosenOrder == freqOrder:
    tag = 'FREQ'

coef_tstats=stats.ttest_1samp(allcoefs, 0, axis=3)

for lag in range(nlags):
    fig,ax = plt.subplots(figsize=[20,15])
    sns.heatmap(coef_tstats.pvalue[lag],ax=ax, cmap='YlGnBu', xticklabels=chosenOrder, yticklabels=chosenOrder, vmin=0, vmax=0.1)
    ax.axes.set_title('Mean Pvalues Lag {}'.format(lags[lag]), fontsize=45)
    ax.tick_params(labelsize=7)
    plt.savefig('./results/ridge_regression/figs/extendedLags/{}_Lag{}_meanPvals.pdf'.format(tag, lags[lag]))

mn_allcoefs=np.mean(allcoefs,axis=3)
for lag in range(nlags):
    fig,ax = plt.subplots(figsize=[20,15])
    sns.heatmap(mn_allcoefs[lag],ax=ax, cmap='seismic', xticklabels=chosenOrder, yticklabels=chosenOrder, vmin=-0.1, vmax=0.1)
    ax.axes.set_title('Mean Betas Lag {}'.format(lags[lag]), fontsize=45)
    ax.tick_params(labelsize=7)
    plt.savefig('./results/ridge_regression/figs/extendedLags/{}_Lag{}_meanCoefs.pdf'.format(tag,lags[lag]))

for lag in range(nlags):        
    pvals = coef_tstats.pvalue[lag]
    sigPval = np.zeros((150,150))
    sigs = np.where(pvals < 0.01)
    sigs = list(zip(sigs[0], sigs[1]))
    for coord in sigs:
        sigPval[coord[0]][coord[1]] = 1

    fig,ax = plt.subplots(figsize=[20,15])
    sns.heatmap(sigPval,ax=ax, cmap='binary', xticklabels=chosenOrder, yticklabels=chosenOrder, vmin=0, vmax=1)
    ax.axes.set_title('Pairs with p < 0.01, Lag {}'.format(lags[lag]), fontsize=45)
    ax.tick_params(labelsize=7)
    plt.savefig('./results/ridge_regression/figs/extendedLags{}_Lag{}_meanSigs.pdf'.format(tag,lags[lag]))
"""
