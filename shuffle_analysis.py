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
    count = 0
    while count < 100000000:
        idx = range(len(lst))
        i1, i2 = random.sample(idx, 2)
        lst[i1], lst[i2] = lst[i2], lst[i1]
        count += 1
    return lst

def get_matrix(lst, counts_dict, mapping, order, X, realpth, controlpth):
    one_hot_items = freq.one_hot(lst)
    pool_count = freq.pooled_frequent_items(one_hot_items, counts_dict)
    lev_df = freq.create_leverage_matrix(lst, pool_count, mapping)
    freq.plot_matrix(lev_df, order, outpath=realpth)
    freq.self_cluster(lev_df, controlpth)

if __name__ == "__main__":
    
    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)
    print('There are {} baskets'.format(len(itemsets)))

    with open('single_counts.pickle', 'rb') as f:
        counts = pickle.load(f)

    with open('mapping.pickle', 'rb') as f:
        maps = pickle.load(f)
    
    with open('lch_order.pickle', 'rb') as f:
        order = pickle.load(f)

    X = 150

    #shuffle each 200 ms basket and then pool across baskets
    shuffle_pooled = []
    shuffled = shuffle_baskets(itemsets)
    shuffle_pooled.append(shuffled)
    for pool in range(4,11,3):
        a = freq.pool_baskets(shuffled, pool)
        shuffle_pooled.append(a)

    for i in range(4):
        realpath = './results/figures/shuffled/basket_shuffle_leverage_matrix_{}.pdf'.format(i)
        controlpath = './results/figures/shuffled/basket_shuffle_leverage_matrix_{}_levorder.pdf'.format(i)
        get_matrix(shuffle_pooled[i], counts, maps, order, X, realpath, controlpath)
    
    pooled = []
    items_shuffled = shuffle_items(itemsets)
    pooled.append(items_shuffled)
    for pool in range(4,11,3):
        a = freq.pool_baskets(items_shuffled, pool)
        pooled.append(a)
    
    for i in range(4):
        realpath = './results/figures/shuffled/item_shuffle_leverage_matrix_{}.pdf'.format(i)
        controlpath = './results/figures/shuffled/item_shuffle_leverage_matrix_{}_levorder.pdf'.format(i)
        get_matrix(pooled[i], counts, maps, order, X, realpath, controlpath)