import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy import stats

with open('./results/ridge_regression/LCH_allcoefs_extendedlags.pickle', 'rb') as f:
    allcoefs = pickle.load(f)

with open('lch_order.pickle', 'rb') as f:
    lchOrder = pickle.load(f)

with open('freq_order.pickle', 'rb') as f:
    freqOrder = pickle.load(f)

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
