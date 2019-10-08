import pickle
import copy
import random
import movie_analysis_v2
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def shuffle_items(lst):
    """
    Randomly shuffle items between baskets 1 million times, ensuring no repetition of an item in a basket.
    lst: the itemsets, either pooled or not.
    """
    _ = copy.deepcopy(lst)
    count = 0
    while count < 1000000:
        a = random.choice(_)
        b = random.choice(_)
        if not a == b:
            rand_idx_a = random.randint(0, len(a)-1)
            rand_idx_b = random.randint(0, len(b)-1)
            if not a[rand_idx_a] in b:
                a[rand_idx_a], b[rand_idx_b] = b[rand_idx_b], a[rand_idx_a]
                count += 1
    return _

def shuffle_baskets(lst):
    """
    Shuffle the basket order rather than items within the baskets.
    """
    _ = copy.deepcopy(lst)
    #_ = lst
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

    #shuffle order of baskets
    bask_shuffled = shuffle_baskets(itemsets)

    #pool these
    shuffle_bask_pooled = []
    for pool in range(1,11,3):
        pooled_itemsets = movie_analysis_v2.pool_baskets(bask_shuffled, pool)
        shuffle_bask_pooled.append(pooled_itemsets)
        print('For the {} group there are {} number of baskets'.format(pool,len(pooled_itemsets)))
    
    #perform apriori on the shuffled basket data
    one_support = 0.021
    four_support = 0.075
    seven_suport = 0.12
    ten_support = 0.163

    movie_analysis_v2.perform_apriori_association(itemsets=shuffle_bask_pooled[0], min_sup=one_support, itemsets_path='./results/frequent_itemsets/correct_basket_shuffle_itemsets_1.csv', rules_path='./results/association_rules/correct_basket_shuffle_association_rules_1.csv')
    movie_analysis_v2.perform_apriori_association(itemsets=shuffle_bask_pooled[1], min_sup=four_support, itemsets_path='./results/frequent_itemsets/correct_basket_shuffle_itemsets_4.csv', rules_path='./results/association_rules/correct_basket_shuffle_association_rules_4.csv')
    movie_analysis_v2.perform_apriori_association(itemsets=shuffle_bask_pooled[2], min_sup=seven_suport, itemsets_path='./results/frequent_itemsets/correct_basket_shuffle_itemsets_7.csv', rules_path='./results/association_rules/correct_basket_shuffle_association_rules_7.csv')
    movie_analysis_v2.perform_apriori_association(itemsets=shuffle_bask_pooled[3], min_sup=ten_support, itemsets_path='./results/frequent_itemsets/correct_basket_shuffle_itemsets_10.csv', rules_path='./results/association_rules/correct_basket_shuffle_association_rules_10.csv')

    #shuffle items and pool those
    if itemsets == bask_shuffled:
        print('Error: the itemsets are not controlled')
    else:
        shuffle_items_pooled = []
        for pool in range(1,11,3):
            pooled_itemsets = movie_analysis_v2.pool_baskets(itemsets, pool)
            shuffle_items_pooled.append(pooled_itemsets)
            print('For the {} group there are {} number of baskets'.format(pool,len(pooled_itemsets)))
        for i in shuffle_items_pooled:
            i = shuffle_items(i)
        
    if shuffle_items_pooled[0] == itemsets:
        print('Error: shuffling items did not work')
    else:
        movie_analysis_v2.perform_apriori_association(itemsets=shuffle_items_pooled[0], min_sup=one_support, itemsets_path='/results/frequent_itemsets/item_shuffle_itemsets_1.csv', rules_path='./results/association_rules/item_shuffle_association_rules_1.csv')
        movie_analysis_v2.perform_apriori_association(itemsets=shuffle_items_pooled[1], min_sup=one_support, itemsets_path='/results/frequent_itemsets/item_shuffle_itemsets_4.csv', rules_path='./results/association_rules/item_shuffle_association_rules_4.csv')
        movie_analysis_v2.perform_apriori_association(itemsets=shuffle_items_pooled[2], min_sup=one_support, itemsets_path='/results/frequent_itemsets/item_shuffle_itemsets_7.csv', rules_path='./results/association_rules/item_shuffle_association_rules_7.csv')
        movie_analysis_v2.perform_apriori_association(itemsets=shuffle_items_pooled[3], min_sup=one_support, itemsets_path='/results/frequent_itemsets/item_shuffle_itemsets_10.csv', rules_path='./results/association_rules/item_shuffle_association_rules_10.csv')