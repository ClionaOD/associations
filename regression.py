import random
import numpy as np
import pandas as pd
import pickle
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskLassoCV

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
    
    Diags = False
    offDiags = True
    sweeps = np.linspace(1,36000,num=40, dtype=int)
    nitems=150
    
    lags = [1,4500,13500,22500,31500,40500]
    nlags = len(lags)

    if Diags == True:
        diags = np.zeros((nitems, len(sweeps)))
        arr = one_hot_enc(itemsets, chosenOrder)
        for lag in range(len(sweeps)):
            y = arr[sweeps[-1]:,:]
            X = arr[sweeps[-1] - sweeps[lag] : -sweeps[lag], :]
            coef = ridge_regress(X,y)
            d = coef.diagonal()
            diags[:,lag] = d
        
        df = pd.DataFrame(diags,columns=sweeps,index=chosenOrder)

        with open('./results/ridge_regression/diagonals_linspace.pickle', 'wb') as f:
            pickle.dump(df,f)
    
    elif offDiags == True:
        offdiags = np.zeros(((nitems*nitems)-nitems, len(sweeps),len(div_itemsets)))

        for i in range(len(div_itemsets)):
            arr = one_hot_enc(div_itemsets[i],chosenOrder)
            div_offd = np.zeros(((nitems*nitems)-nitems, len(sweeps)))

            for lag in range(len(sweeps)):
                y = arr[sweeps[-1]:,:]
                X = arr[sweeps[-1] - sweeps[lag] : -sweeps[lag], :]
                coef = ridge_regress(X,y)

                #remove diagonals and flatten
                offd = coef[~np.eye(coef.shape[0],dtype=bool)].reshape(coef.shape[0],-1)
                offd = offd.reshape(-1)
                div_offd[:,lag] = offd

            offdiags[:,:,i] = div_offd
        
        with open('./results/ridge_regression/divided_offDiagonals_linspace.pickle','wb') as f:
            pickle.dump(offdiags,f)
            
    else:
        allcoefs = np.zeros((nlags,nitems,nitems,len(div_itemsets)))

        for i in range(0, len(div_itemsets)):
            arr = one_hot_enc(div_itemsets[i], chosenOrder)
            div_coefs = np.zeros((nlags, nitems, nitems))
            
            for lag in range(nlags):
                y = arr[lags[-1]:,:]
                X = arr[lags[-1] - lags[lag] : -lags[lag], :]
                coef = ridge_regress(X,y)
                div_coefs[lag,:,:] = coef
            
            allcoefs[:,:,:,i] = div_coefs
        
        with open('./results/ridge_regression/LCH_allcoefs_extendedlags.pickle', 'wb') as f:
            pickle.dump(allcoefs,f)

    