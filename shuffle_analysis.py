"""
Author: Cliona O'Doherty
Description: Continue the analysis as in analyse_itemsets.py, this time with controls for 
shuffling the temporal order of the baskets.
"""
import pickle
import copy
import random
import get_frequent_items as freq
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def shuffle_items(lst):
    """
    Randomly shuffle items between baskets 100 million times, ensuring no repetition of an item in a basket.
    lst: the itemsets, either pooled or not.
    """
    count = 0
    while count < 100000000:
        a = random.choice(lst)
        b = random.choice(lst)
        if not a == b:
            rand_idx_a = random.randint(0, len(a)-1)
            rand_idx_b = random.randint(0, len(b)-1)
            if not a[rand_idx_a] in b:
                a[rand_idx_a], b[rand_idx_b] = b[rand_idx_b], a[rand_idx_a]
                count += 1
    return lst

def shuffle_baskets(lst):
    """
    Shuffle the basket order rather than items within the baskets.
    """
    _ = copy.deepcopy(lst)
    count = 0
    while count < 1000000:
        idx = range(len(_))
        i1, i2 = random.sample(idx, 2)
        _[i1], _[i2] = _[i2], _[i1]
        count += 1
    return _

def get_matrix(lst,X):
    
    one_hot_items = freq.one_hot(lst)
    single_counts, mapping = freq.most_frequent_items(one_hot_items, X)
    lev_array = freq.create_leverage_matrix(lst, single_counts, mapping)
    lev_df = freq.order_matrix(lev_array, mapping, X)
    return lev_df

if __name__ == "__main__":
    
    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)
    print('There are {} baskets'.format(len(itemsets)))

    X = 150

    #shuffle each 200 ms basket and then pool across baskets
    shuffled = shuffle_baskets(itemsets)
    shuffle_pooled = []
    for pool in range(1,11,3):
        a = freq.pool_baskets(shuffled, pool)
        shuffle_pooled.append(a)

    for i in range(4):
        outpath = './results/figures/v4/basket_shuffle_leverage_matrix_{}.pdf'.format(i)
        freq.plot_matrix(get_matrix(shuffle_pooled[i], outpath), X)
    
    items_shuffled = shuffle_items(itemsets)
    pooled = []
    for pool in range(1,11,3):
        a = freq.pool_baskets(items_shuffled, pool)
        pooled.append(a)
    
    for i in range(4):
        outpath = './results/figures/v4/item_shuffle_leverage_matrix_{}.pdf'.format(i)
        freq.plot_matrix(get_matrix(pooled[i], outpath), X)