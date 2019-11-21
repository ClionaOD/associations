import random
import numpy as np
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

def perform_var(lst, nlags):
    te = TransactionEncoder()
    one_hot = te.fit(lst).transform(lst, sparse=False)
    one_hot = one_hot.astype(int)

    model = VAR(one_hot)
    results = model.fit(maxlags=nlags)
    print(results.summary())

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    div_itemsets = divide_dataset(itemsets, 8)

    for i in div_itemsets:
        perform_var(i, 1)