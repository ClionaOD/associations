import random
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.api import VAR
from mlxtend.preprocessing import TransactionEncoder

def divide_dataset(lst, div):
    length = int(len(lst)/div)

    split_lst = []
    mult = 0
    for i in range(div):
        x = lst[length*mult : length*(mult+1)]
        split_lst.append(x)
        mult += 1

    return split_lst

def most_freq_one_hot(lst, X=150):
    te = TransactionEncoder()
    one_hot = te.fit(lst).transform(lst, sparse=False)
    one_hot = one_hot.astype(int)
    one_hot_df = pd.DataFrame(one_hot, columns=te.columns_)

    counts = one_hot_df.sum(axis=0, skipna=True)
    top_X = pd.DataFrame(counts.nlargest(X,  keep='all'))
    

    freq_items = top_X.index.tolist()
    freq_onehot = one_hot_df[freq_items]

    onehot_arr = freq_onehot.values
    return onehot_arr


def perform_var(arr, nlags, div):
    model = VAR(arr)
    results = model.fit(maxlags=nlags)

    out_coefs = results.coefs
    
    for i in range(4):
        coef_path = './results/var/coefs/coef_array_{}_{}.txt'.format(div, i)
        np.savetxt(coef_path, out_coefs[i])

    out_tvals = results.tvalues
    tval_path = './results/var/tvalues/tval_array_{}.txt'.format(div)
    np.savetxt(tval_path, out_tvals)

    out_pvals = results.pvalues
    pval_path = './results/var/pvalues/pval_array_{}.txt'.format(div)
    np.savetxt(pval_path, out_pvals)

    return results

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    div_itemsets = divide_dataset(itemsets, 8)

    for i in range(0,len(div_itemsets)):
        arr = most_freq_one_hot(div_itemsets[i])
        perform_var(arr, nlags=4, div=i)

        