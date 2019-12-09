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
    

"""
def perform_var(arr, nlags, div):
    model = VAR(arr)
    results = model.fit(maxlags=nlags)

    out_coefs = results.coefs
    
    fig,ax=plt.subplots(ncols=nlags)
    if nlags==1:
        ax=[ax]
    for lag in range(nlags):
        sns.heatmap(out_coefs[lag],ax=ax[lag])

    plt.savefig('./results/var/coefs/all_coefs_{}.pdf'.format(div))
    plt.close()

    return results, out_coefs
"""

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    nitems=150

    frequent_items = most_freq(itemsets, X=nitems)

    div_itemsets = divide_dataset(itemsets, 16)

    nlags=4
    decimateby=5

    allcoefs = np.zeros((nlags,nitems,nitems,len(div_itemsets)))
    
    for i in range(0, len(div_itemsets)):
        arr = one_hot_enc(div_itemsets[i], frequent_items)
        nobs = len(arr)
        div_coefs = np.zeros((nlags, nitems, nitems))
        
        if decimateby:
            arr = signal.decimate(arr,decimateby,axis=0)
        
        for lag in range(1,nlags+1):
            y = arr[nlags:,:]
            X = arr[nlags-lag:-lag,:]
            coef = ridge_regress(X,y)
            div_coefs[lag-1,:,:] = coef
        
        fig,ax = plt.subplots(ncols=nlags)
        if nlags == 1:
            ax = [ax]
        for lag in range(nlags):
            sns.heatmap(div_coefs[lag], ax=ax[lag])
        plt.savefig('./results/ridge_regression/coefs/coefs_div{}.pdf'.format(i))
        plt.close()
        
        allcoefs[:,:,:,i] = div_coefs

    """

    allcoefs=np.zeros((nlags,nitems,nitems,len(div_itemsets)))
    for i in range(0,len(div_itemsets)):
        arr = one_hot_enc(div_itemsets[i], frequent_items)
        if decimateby:
            arr=signal.decimate(arr,decimateby,axis=0)
            #plt.figure()
            #sns.heatmap(arr)
            #plt.figure()
            #plt.plot(arr[1:500,:])
        results, coefs = perform_var(arr, nlags=nlags, div=i)
        allcoefs[:,:,:,i]=coefs
    """

    coef_tstats=stats.ttest_1samp(allcoefs, 0, axis=3)

    fig,ax=plt.subplots(ncols=nlags, figsize=[12,8])
    if nlags==1:
        ax=[ax] 
    for lag in range(nlags):
        sns.heatmap(coef_tstats.pvalue[lag],ax=ax[lag], cmap='BuPu', vmin=0, vmax=0.1)
    plt.savefig('./results/ridge_regression/mean_tstats_pvals_{}lags.pdf'.format(nlags))

    mn_allcoefs=np.mean(allcoefs,axis=3)
    fig,ax=plt.subplots(ncols=nlags, figsize=[12,8])
    if nlags==1:
        ax=[ax] 
    for lag in range(nlags):
        sns.heatmap(mn_allcoefs[lag],ax=ax[lag], vmin=-1, vmax=1, cmap='seismic')
    plt.savefig('./results/ridge_regression/mean_coefs_{}lags.pdf'.format(nlags))

    plt.show()


    