import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from itertools import combinations

def one_hot(lst): 
    """one hot encode lst, a list of lists"""
    te = TransactionEncoder()
    te_ary = te.fit(lst).transform(lst, sparse=True)
    df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)
    return df

def most_frequent_items(one_hot_df, X=150):
    """
    get top X (default 150) most frequent items from a one-hot encoded DataFrame
    store this is an encoded dictionary mapping each unique entry to an integer
    
    returns: 
    single_counts: a dict with each item and its count
    mapping: a dict with the numeric values of each unique item 
    """
    one_hot_counts = one_hot_df.sum(axis=0, skipna=True)
    _ = one_hot_counts.to_dict()
    wn_counts = {item : freq for item, freq in _.items()if len(wn.synsets(item, pos='n')) != 0}
    wn_counts = pd.DataFrame.from_dict(wn_counts, orient='index')

    top_x = wn_counts.nlargest(X, columns=0, keep='all') 
    single_counts = top_x[0].to_dict()
    keys = list(single_counts.keys())
    mapping = {k:v for v,k in enumerate(keys)}

    return single_counts, mapping

def pooled_frequent_items(one_hot_df, counts_dict):
    items = list(counts_dict.keys())
    one_hot_counts = one_hot_df.sum(axis=0, skipna=True)
    _ = one_hot_counts.to_dict()

    pooled_counts = {item: freq for item, freq in _.items() if item in items}
    return pooled_counts

def get_lch_order(items_dict, synset_mapping, matrix_outpath, dend_outpath):
    """ 
    get LCH distance for most frequent items and order them by hierarchical clustering, returning the order of labels 
    items_dict: dictionary with keys top X most frequent items and values their frequency
    synset_mapping: dictionary with most frequent items and their correct synsets

    returns: a list of items to order the matrices by (acc. to LCH)
    """

    items = list(items_dict.keys())
    synsets_list = []
    for k in items:
        syn = synset_mapping[k]
        synsets_list.append(syn)
    synsets_list = [wn.synset(i) for i in synsets_list]

    cmb = list(combinations(synsets_list,2))
    lch_list = []
    x = []
    y = []
    for item in cmb:
        x.append(item[0])
        y.append(item[1])
        lch = item[0].lch_similarity(item[1])
        lch_list.append(lch)

    x = np.array(x,dtype = str)
    y = np.array(y, dtype = str)
    lch_list = np.array(lch_list,dtype = float)

    lch_matrix = squareform(lch_list)
    d = synsets_list[0].lch_similarity(synsets_list[0])
    np.fill_diagonal(lch_matrix, d)

    arr = lch_matrix - lch_matrix.mean(axis=0)
    arr = arr / np.abs(arr).max(axis=0)

    fig,ax = plt.subplots(figsize=(10,10))
    dend = dendrogram(linkage(arr, method='ward'), ax=ax, labels=items)
    ax.tick_params(axis='x', labelsize=4)
    plt.savefig(dend_outpath)
    plt.close()

    orderedNames = dend['ivl']

    df = pd.DataFrame(data=arr, index=items, columns=items)
    df = df.reindex(orderedNames, columns=orderedNames)   
    cmap = plt.cm.coolwarm
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(df, ax=ax, cmap=cmap)
    plt.savefig(matrix_outpath)
    plt.close()

    return orderedNames

def create_leverage_matrix(itemsets, counts_dict, mapping):
    """
    itemsets: list of baskets, either pooled or not
    counts_dict: a dictionary of each of the most frequent items and their frequency
    mapping: the integer value mapping for each of the frequent items
    """
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

    return leverage

def order_matrix(array, mapping, lch_order):
    lch_encoded = [mapping[k] for k in lch_order]

    df = pd.DataFrame(array)
    df = df.reindex(lch_encoded, columns=lch_encoded)
    df.columns = lch_order
    df.index = lch_order
    return df

def plot_matrix(df, outpath):
    cmap = plt.cm.coolwarm
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(df, ax=ax, cmap=cmap)
    plt.savefig(outpath)
    plt.close()

def pool_baskets(inlist, multiply_frames=1):
    """
    Pool the basket latencies from default 200 ms up to desired length.
    inlist: the itemsets, a list of lists containing the labels
    multiply_frames: what to multiply 200 ms by to get desired pooling e.g. multiply_frames=10 for 2 second baskets
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

if __name__ == "__main__":

    with open('itemsets.pickle', 'rb') as f:
        itemsets = pickle.load(f)
    
    with open('item_synsets.pickle', 'rb') as f:
        item_synsets = pickle.load(f)

    X = 150
    
    one_hot_items = one_hot(itemsets)
    single_counts, mapping = most_frequent_items(one_hot_items, X)
    
    lch_order = get_lch_order(single_counts, synset_mapping=item_synsets, dend_outpath='./results/figures/v4/dendrogram.pdf', matrix_outpath='./results/figures/v4/lch_matrix.pdf')
    
    lev_array = create_leverage_matrix(itemsets, single_counts, mapping)
    lev_df = order_matrix(lev_array, mapping, lch_order)
    outpath = './results/figures/v4/leverage_matrix_1.pdf'
    plot_matrix(lev_df, outpath)

    #pool baskets into latency 800 ms, 700 ms, 2000 ms)
    pooled = []
    for pool in range(4,11,3):
        pooled_itemsets = pool_baskets(itemsets, pool)
        pooled.append(pooled_itemsets)
        print('For the {} group there are {} number of baskets'.format(pool,len(pooled_itemsets)))

    for i in range(len(pooled)):
        one_hot_items = one_hot(pooled[i])
        pool_count = pooled_frequent_items(one_hot_items, single_counts)
        lev_array = create_leverage_matrix(pooled[i], pool, mapping)
        lev_df = order_matrix(lev_array, mapping, lch_order)
        outpath = './results/figures/v4/leverage_matrix_poolidx{}.pdf'.format(i)
        plot_matrix(lev_df, outpath)