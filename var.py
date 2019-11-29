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


def perform_var(arr, nlags):
    model = VAR(arr)
    results = model.fit(maxlags=nlags)
    print(results.summary())

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    div_itemsets = divide_dataset(itemsets, 8)

    arr = most_freq_one_hot(div_itemsets[0])
    perform_var(arr, nlags=4)