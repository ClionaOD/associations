import random
import numpy as np
import pandas as pd
import pickle
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    clf = Ridge(alpha=1, normalize=True)
    clf.fit(X,y)
    coefs = clf.coef_
    return coefs

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    with open('lch_order.pickle', 'rb') as f:
       lchOrder = pickle.load(f)

    with open('freq_order.pickle', 'rb') as f:
        frequent_items = pickle.load(f)

    chosenOrder = lchOrder

    div_itemsets = divide_dataset(itemsets, 16)

    nitems=150
    nlags=2
    aggregby = 18000

    allcoefs = np.zeros((nlags,nitems,nitems,len(div_itemsets)))

    for i in range(0, len(div_itemsets)):
        arr = one_hot_enc(div_itemsets[i], chosenOrder)
        div_coefs = np.zeros((nlags, nitems, nitems))
        
        count = 1
        for lag in range(0,nlags*aggregby,aggregby):
            y = arr[nlags*aggregby:,:]
            X = arr[lag:-nlags*aggregby+lag,:] #this puts the lags backwards
            coef = ridge_regress(X,y)
            div_coefs[-count,:,:] = coef #therefore do this (minus index)
            count += 1
        
        allcoefs[:,:,:,i] = div_coefs
    
    with open('LCH_allcoefs_36000lag.pickle', 'wb') as f:
        pickle.dump(allcoefs,f)

    