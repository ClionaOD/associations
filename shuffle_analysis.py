"""
Author: Cliona O'Doherty
Description: Continue the analysis as in analyse_itemsets.py, this time with controls for 
shuffling the temporal order of the baskets.
"""
import pickle
import json
import copy
import random
import get_frequent_items as freq
import pandas as pd

def shuffle_items(lst, mapping):
    """
    Randomly shuffle items between baskets 100 million times, ensuring no repetition of an item in a basket.
    lst: the itemsets, either pooled or not.
    mapping: index values for each item from freq
    """

    clipped_lst = [[i for i in basket if i in list(mapping.keys())] for basket in lst]
    clipped_lst = [i for i in clipped_lst if not len(i) == 0]
    encoded_lst = [[mapping[k] for k in bask] for bask in clipped_lst]

    count = 0
    log = 0
    while count < 50000000:
        a = random.choice(encoded_lst)
        b = random.choice(encoded_lst)
        rand_idx_a = random.randint(0, len(a)-1)
        rand_idx_b = random.randint(0, len(b)-1)
        if not a[rand_idx_a] in b and not b[rand_idx_b] in a:
            a[rand_idx_a], b[rand_idx_b] = b[rand_idx_b], a[rand_idx_a]
            count += 1
        
        if count % 1000000 == 0:
            log += 1
            print('{} million shuffles are complete'.format(log))

        if log > 50:
            count = 50000001

    return_strings = {v: k for k,v in mapping.items()}
    new_lst = [[return_strings[k] for k in encoded_bask] for encoded_bask in encoded_lst]
    
    return new_lst

def get_matrix(lst, counts_dict, mapping, order, X, realpth, controlpth):
    one_hot_items = freq.one_hot(lst)
    pool_count = freq.pooled_frequent_items(one_hot_items, counts_dict)
    lev_df = freq.create_leverage_matrix(lst, pool_count, mapping, X)
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
    shuffled = random.sample(itemsets, (len(itemsets)))
    print('Basket shuffling complete.')
    shuffle_pooled.append(shuffled)
    for pool in range(4,11,3):
        a = freq.pool_baskets(shuffled, pool)
        shuffle_pooled.append(a)


    for i in range(4):
        realpath = './results/figures/shuffled/basket/basket_shuffle_leverage_matrix_{}.pdf'.format(i)
        controlpath = './results/figures/shuffled//basket/basket_shuffle_leverage_matrix_{}_levorder.pdf'.format(i)
        get_matrix(lst=shuffle_pooled[i], counts_dict=counts, mapping=maps, order=order, X=X, realpth=realpath, controlpth=controlpath)
        print('Pool number {} complete.'.format(i))
    
    pooled = []
    items_shuffled = shuffle_items(itemsets, maps)
    print('Item shuffling complete.')
    pooled.append(items_shuffled)
    for pool in range(4,11,3):
        a = freq.pool_baskets(items_shuffled, pool)
        pooled.append(a)
    
    for i in range(4):
        realpath = './results/figures/shuffled/item/item_shuffle_leverage_matrix_{}.pdf'.format(i)
        controlpath = './results/figures/shuffled/item/item_shuffle_leverage_matrix_{}_levorder.pdf'.format(i)
        get_matrix(lst=items_shuffled, counts_dict=counts, mapping=maps, order=order, X=X, realpth=realpath, controlpth=controlpath)