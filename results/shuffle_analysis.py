"""
Author: Cliona O'Doherty
Description: Continue the analysis as in analyse_itemsets.py, this time with controls for 
shuffling the temporal order of the baskets.
"""
import pickle
import copy
import random
import analyse_itemsets
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def shuffle_items(lst):
    """
    Randomly shuffle items between baskets 1 million times, ensuring no repetition of an item in a basket.
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

if __name__ == "__main__":
    
    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)
    print('There are {} baskets in total for basket shuffling'.format(len(itemsets)))

    one_support = 0.021
    four_support = 0.075
    seven_support = 0.12
    ten_support = 0.163

    #This is the control for order of baskets with same contents of items
    pooled_first = []
    for pool in range(1,11,3):
        a = analyse_itemsets.pool_baskets(itemsets, pool)
        pooled_first.append(a)
    for i in pooled_first:
        i = shuffle_baskets(i) 
    analyse_itemsets.perform_apriori_association(itemsets=pooled_first[3], min_sup=ten_support, itemsets_path='./results/frequent_itemsets/wrong_basket_shuffle_itemsets_10.csv', rules_path='./results/association_rules/wrong_basket_shuffle_association_rules_10.csv')

    #this is for pooling having already shuffled, i.e. temporal info is lost from 200 ms to 2000 ms
    shuffled = shuffle_baskets(itemsets)
    shuffled_first = []
    for pool in range(1,11,3):
        b = analyse_itemsets.pool_baskets(shuffled, pool)
        shuffled_first.append(b)
    analyse_itemsets.perform_apriori_association(itemsets=shuffled_first[3], min_sup=ten_support, itemsets_path='./results/frequent_itemsets/basket_shuffle_itemsets_10.csv', rules_path='./results/association_rules/basket_shuffle_association_rules_10.csv')

    a = shuffle_items(itemsets)

one_support = 0.021
four_support = 0.075
seven_support = 0.12
ten_support = 0.163
items_shuffled = []
for pool in range(1,11,3):
    b = analyse_itemsets.pool_baskets(a, pool)
    items_shuffled.append(b)
analyse_itemsets.perform_apriori_association(itemsets=items_shuffled[0], min_sup=one_support, itemsets_path='./results/frequent_itemsets/item_shuffle_itemsets_1.csv', rules_path='./results/association_rules/item_shuffle_association_rules_1.csv')
analyse_itemsets.perform_apriori_association(itemsets=items_shuffled[1], min_sup=four_support, itemsets_path='./results/frequent_itemsets/item_shuffle_itemsets_4.csv', rules_path='./results/association_rules/item_shuffle_association_rules_4.csv')
analyse_itemsets.perform_apriori_association(itemsets=items_shuffled[2], min_sup=seven_support, itemsets_path='./results/frequent_itemsets/item_shuffle_itemsets_7.csv', rules_path='./results/association_rules/item_shuffle_association_rules_7.csv')
analyse_itemsets.perform_apriori_association(itemsets=items_shuffled[3], min_sup=ten_support, itemsets_path='./results/frequent_itemsets/item_shuffle_itemsets_10.csv', rules_path='./results/association_rules/item_shuffle_association_rules_10.csv')