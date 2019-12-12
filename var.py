import random
import numpy as np
import pandas as pd
import pickle
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal
from sklearn.linear_model import Ridge

def divide_dataset(lst, div):
    length = int(len(lst)/div)

    split_lst = []
    mult = 0
    for i in range(div):
        x = lst[length*mult : length*(mult+1)]
        split_lst.append(x)
        mult += 1

    return split_lst

def most_freq(lst, X=150):
    te = TransactionEncoder()
    one_hot = te.fit(lst).transform(lst, sparse=False)
    one_hot = one_hot.astype(int)
    one_hot_df = pd.DataFrame(one_hot, columns=te.columns_)

    counts = one_hot_df.sum(axis=0, skipna=True)
    top_X = pd.DataFrame(counts.nlargest(X,  keep='all'))

    freq_items = top_X.index.tolist()
    return freq_items 

def one_hot_enc(lst, items):
    te = TransactionEncoder()
    one_hot = te.fit(lst).transform(lst, sparse=False)
    one_hot = one_hot.astype(int)
    one_hot_df = pd.DataFrame(one_hot, columns=te.columns_)
    freq_onehot = one_hot_df[items]

    onehot_arr = freq_onehot.values
    items = list(freq_onehot.columns)
    itemMaps = {k:v for k,v in enumerate(items)}
    return onehot_arr

def ridge_regress(X,y):
    clf = Ridge(alpha=1,fit_intercept=False)
    clf.fit(X,y)
    coefs = clf.coef_
    return coefs

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    #with open('lch_order.pickle', 'rb') as f:
     #   lchOrder = pickle.load(f)

    nitems=150

    frequent_items = most_freq(itemsets, X=nitems)
    with open('freq_order.pickle', 'wb') as f:
        pickle.dump(frequent_items, f)

    chosenOrder = lchOrder

    div_itemsets = divide_dataset(itemsets, 16)

    nlags=4
    decimateby=False
    aggregby = 5

    allcoefs = np.zeros((nlags,nitems,nitems,len(div_itemsets)))

    for i in range(0, len(div_itemsets)):
        arr = one_hot_enc(div_itemsets[i], chosenOrder)
        nobs = len(arr)
        div_coefs = np.zeros((nlags, nitems, nitems))
        
        if decimateby:
            arr = signal.decimate(arr,decimateby,axis=0)
        
        count = 1
        for lag in range(0,nlags*aggregby,aggregby):
            y = arr[nlags*aggregby:,:]
            X = arr[lag:-nlags*aggregby+lag,:] #this puts the lags backwards
            coef = ridge_regress(X,y)
            div_coefs[-count,:,:] = coef #therefore do this (minus index)
            count += 1
        
        allcoefs[:,:,:,i] = div_coefs

    coef_tstats=stats.ttest_1samp(allcoefs, 0, axis=3)
    maps = {str(k):v for k,v in enumerate(chosenOrder)}

    if chosenOrder == frequent_items:
        tag = 'FREQ'
    elif chosenOrder == lchOrder:
        tag = 'LCH'

    fig,ax=plt.subplots(ncols=nlags, figsize=[30,10])
    if nlags==1:
        ax=[ax] 
    for lag in range(nlags):
        if lag == range(nlags)[-1]:
            cbar=True
        else:
            cbar=False
        sns.heatmap(coef_tstats.pvalue[lag],ax=ax[lag], cbar=cbar, cmap='YlGnBu', vmin=0, vmax=0.1)
        xlabels = ax[lag].xaxis.get_ticklabels()
        xlabels = [k.get_text() for k in xlabels]
        ylabels = ax[lag].yaxis.get_ticklabels()
        ylabels = [k.get_text() for k in ylabels]

        newlabels_x = [maps[k] for k in xlabels]
        newlabels_y = [maps[k] for k in ylabels]

        ax[lag].set_xticklabels(newlabels_x)
        ax[lag].set_yticklabels(newlabels_y)

    plt.savefig('./results/ridge_regression/{}_undecMean_pvals.pdf'.format(tag))

    mn_allcoefs=np.mean(allcoefs,axis=3)
    fig,ax=plt.subplots(ncols=nlags, figsize=[30,10])
    if nlags==1:
        ax=[ax] 
    for lag in range(nlags):
        if lag == range(nlags)[-1]:
            cbar=True
        else:
            cbar=False
        sns.heatmap(mn_allcoefs[lag],ax=ax[lag], cbar=cbar, vmin=-0.1, vmax=0.1, cmap='seismic')
        xlabels = ax[lag].xaxis.get_ticklabels()
        xlabels = [k.get_text() for k in xlabels]
        ylabels = ax[lag].yaxis.get_ticklabels()
        ylabels = [k.get_text() for k in ylabels]

        newlabels_x = [maps[k] for k in xlabels]
        newlabels_y = [maps[k] for k in ylabels]

        ax[lag].set_xticklabels(newlabels_x)
        ax[lag].set_yticklabels(newlabels_y)
    plt.savefig('./results/ridge_regression/{}_undecMean_coefs.pdf'.format(tag))

    fig,ax = plt.subplots(ncols=nlags, figsize=[30,10])
    if nlags==1:
        ax=[ax]
    for lag in range(nlags):
        
        pvals = coef_tstats.pvalue[lag]
        sigPval = np.zeros((150,150))
        sigs = np.where(pvals < 0.01)
        sigs = list(zip(sigs[0], sigs[1]))
        for coord in sigs:
            sigPval[coord[0]][coord[1]] = 1

        if lag == range(nlags)[-1]:
            cbar=True
        else:
            cbar=False
        sns.heatmap(sigPval, ax=ax[lag], cbar=cbar, cmap='binary', vmin=0, vmax=1)
        xlabels = ax[lag].xaxis.get_ticklabels()
        xlabels = [k.get_text() for k in xlabels]
        ylabels = ax[lag].yaxis.get_ticklabels()
        ylabels = [k.get_text() for k in ylabels]

        newlabels_x = [maps[k] for k in xlabels]
        newlabels_y = [maps[k] for k in ylabels]

        ax[lag].set_xticklabels(newlabels_x)
        ax[lag].set_yticklabels(newlabels_y)
    plt.savefig('./results/ridge_regression/{}_undecMeanPval_P<0.01.pdf'.format(tag))

    plt.show()


    