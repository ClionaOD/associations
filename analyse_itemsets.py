"""
Author: Cliona O'Doherty

Description: A script for the analysis of labels returned from Amazon Rekognition video tagging. 
This script performs an association analysis on the labels, finding the items that frequently co-occur. 
Various association metrics are returned using mlxtend apriori and association rules
"""

import pickle
import glob
import pandas as pd
import random
import copy
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def create_baskets(dict1):
    """
    Create the empty baskets from the labels returned by Rekognition, one for each unique timestamp.
    """
    uniquetimestamps = set([x['Timestamp'] for x in dict1['alllabels']])
    basket = { i: [] for i in uniquetimestamps}
    return basket

def fill_baskets(dict1, dict2):
    """
    Fills the empty baskets with labels corresponding to the timestamp.
    dict1: the empty baskets for each movie (i.e. basket from create_baskets).
    dict2: the same as arg passed to create_baskets, i.e. each movie's labels.
    """
    for vals in dict2['alllabels']:
        dict1[vals['Timestamp']].extend([vals['Label']['Name']])
    
    return dict1

def pool_baskets(inlist, multiply_frames=1):
    """
    Pool the basket latencies from default 200 ms up to desired length.
    inlist: the itemsets, a list of lists containing the labels
    multiple_frames: what to multiply 200 ms by to get desired pooling e.g. multiply_frames=10 for 2 second baskets
    """
    outlist = []
    startpoint = 0

    for j in range(0,len(inlist)//multiply_frames):
        endpoint = startpoint + multiply_frames
        outelement = []
        list_temp = inlist[startpoint:endpoint]
        for i in range(0,multiply_frames):
            outelement.extend(list_temp[i])
        outlist.append(outelement)
        startpoint = startpoint + multiply_frames

    if len(inlist)%multiply_frames != 0:
        remainder = len(inlist)%multiply_frames
        remainderlist = inlist[-remainder:]
        outelement = []
        for i in range(0,remainder):
            outelement.extend(remainderlist[i])
        outlist.append(outelement)

    final_list =[]
    for i in outlist:
        s = set(i)
        outlist2 = list(s)
        final_list.append(outlist2)
    return final_list

def perform_apriori_association(itemsets, min_sup, itemsets_path, rules_path):
    """
    itemsets: a list of lists containing the baskets/labels.
    min_sup: the minimum support threshold to set for the apriori algorithm.
    itemsets_path: the output path for a .csv file containing the frequent itemsets.
    rules_path: the output path for a .csv file containing association rules and metrics.
    """
    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
    df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True, verbose=1, max_len=2) 
    frequent_itemsets.to_csv(itemsets_path, sep=',', index=False)

    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
    rules = rules[rules.confidence != 1]
    rules.to_csv(rules_path, sep=',', index=False)

if __name__ == "__main__":

    #import labels    
    files = glob.glob('./data/*.pickle')
    movies = []
    for file in files:
        file_Name = file
        fileObject = open(file_Name, 'rb')
        file_labels = pickle.load(fileObject) 
        movies.append(file_labels)

    for movie in movies:
        del movie['compmsg']
        del movie['deltat']
        del movie['vid']

    #structure labels into "baskets" of latency 200 ms
    itemsets = []
    for movie in movies:
        baskets = fill_baskets(create_baskets(movie), movie)
        itemsets.extend(baskets.values())
    with open('itemsets.pickle', 'wb') as f:
        pickle.dump(itemsets, f)

    count = 0 
    for basket in itemsets:
        count += len(basket)
    print('The total number of baskets is {}'.format(len(itemsets)))
    print("The total number of labels is {}".format(count))

"""
    #pool baskets into latency 200 ms, 800 ms, 700 ms, 2000 ms)
    pooled = []
    for pool in range(1,11,3):
        pooled_itemsets = pool_baskets(itemsets, pool)
        pooled.append(pooled_itemsets)
        print('For the {} group there are {} number of baskets'.format(pool,len(pooled_itemsets)))

    #perform apriori and association
    one_support = 0.021
    four_support = 0.075
    seven_suport = 0.12
    ten_support = 0.163

    perform_apriori_association(itemsets=pooled[0], min_sup=one_support, itemsets_path='./results/frequent_itemsets/itemsets_1.csv', rules_path='./results/association_rules/association_rules_1.csv')
    perform_apriori_association(itemsets=pooled[1], min_sup=four_support, itemsets_path='./results/frequent_itemsets/itemsets_4.csv', rules_path='./results/association_rules/association_rules_4.csv')
    perform_apriori_association(itemsets=pooled[2], min_sup=seven_suport, itemsets_path='./results/frequent_itemsets/itemsets_7.csv', rules_path='./results/association_rules/association_rules_7.csv')
    perform_apriori_association(itemsets=pooled[3], min_sup=ten_support, itemsets_path='./results/frequent_itemsets/itemsets_10.csv', rules_path='./results/association_rules/association_rules_10.csv')

    

    