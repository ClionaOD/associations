import pickle
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder

def one_hot(lst): 
    """one hot encode lst, a list of lists"""
    te = TransactionEncoder()
    te_ary = te.fit(lst).transform(lst, sparse=True)
    df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)
    return df

def reorder_od(dict1,order):
    new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
    new_od.update(dict1)
    return new_od

def most_frequent_items(one_hot_df, X=150):
    """
    get top X (default 150) most frequent items from a one-hot encoded DataFrame
    store this is an encoded dictionary mapping each unique entry to an integer
    
    returns: 
    single_counts, a dict with each item and its count
    mapping, a dict with the numeric values of each unique item 
    """
    one_hot_counts = one_hot_df.sum(axis=0, skipna=True)
    top_150 = one_hot_counts.nlargest(X, keep='all')
    single_counts = top_150.to_dict()
    keys = list(single_counts.keys())
    mapping = {k:v for v,k in enumerate(keys)}

    return single_counts, mapping

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

def create_leverage_matrix(itemsets, counts_dict, mapping):

    #first create the X * 1 probability matrix
    single_probs = {k: v/len(itemsets) for k,v in counts_dict.items()}
    encoded = {mapping.get(k, k): v for k,v in single_probs.items()}
    single_probs_df = pd.DataFrame.from_dict(encoded, orient='index')
    single_probs_array = single_probs_df.values

    #clean itemsets for efficiency
    clipped_itemsets = [[i for i in basket if i in list(counts_dict.keys())] for basket in itemsets]
    encoded_itemsets = [[mapping[k] for k in basket] for basket in clipped_itemsets]

    #now create the X*X conditional probability matrix
    pair_counts = np.zeros((150,150))

    for basket in encoded_itemsets:
        for idx, x in enumerate(basket[:-1]):
            for y in basket[idx+1 :]:
                pair_counts[x,y] += 1
    
    pair_probs = pair_counts/len(itemsets)
    pair_probs = pair_probs + np.transpose(pair_probs)

    #leverage = conditional probability - independent probability
    leverage = pair_probs - (np.matmul(single_probs_array, single_probs_array.T))
    leverage = leverage.clip(min=0)

    return leverage

def order_matrix(array, mapping, X):
    reverse_mapping = collections.OrderedDict({v:k for k,v in mapping.items()})
    order = list(range(X))
    reverse_mapping = reorder_od(reverse_mapping, order)
    matrix_order = list(reverse_mapping.values())

    df = pd.DataFrame(array, index=matrix_order, columns=matrix_order)
    return df

def plot_matrix(df, outpath):
    x = sns.clustermap(lev_df, vmin=0, vmax=0.1, center=0.05, figsize=(20,20), method='ward')
    x.savefig(outpath)

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)

    #pool baskets into latency 200 ms, 800 ms, 700 ms, 2000 ms)
    pooled = []
    for pool in range(1,11,3):
        pooled_itemsets = pool_baskets(itemsets, pool)
        pooled.append(pooled_itemsets)
        print('For the {} group there are {} number of baskets'.format(pool,len(pooled_itemsets)))

    X = 150

    for i in range(4):
        one_hot_items = one_hot(pooled[i])
        single_counts, mapping = most_frequent_items(one_hot_items, X)
        lev_array = create_leverage_matrix(itemsets, single_counts, mapping)
        lev_df = order_matrix(lev_array, mapping, X)
        outpath = './results/figures/v4/leverage_matrix_{}.pdf'.format(i)
        plot_matrix(lev_df, outpath)

    