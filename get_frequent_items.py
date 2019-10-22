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
from scipy.spatial.distance import pdist
from itertools import combinations
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/home/CUSACKLAB/clionaodoherty/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

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
    wn_counts = {item : freq for item, freq in _.items() if len(wn.synsets(item, pos='n')) != 0 and item in model.vocab}
    wn_counts = pd.DataFrame.from_dict(wn_counts, orient='index')

    top_x = wn_counts.nlargest(X, columns=0, keep='all') 
    single_counts = top_x[0].to_dict()
    keys = list(single_counts.keys())
    mapping = {k:v for v,k in enumerate(keys)}

    return single_counts, mapping

def pooled_frequent_items(one_hot_df, counts_dict):
    """
    one_hot_df: one hot encoded dataframe for the pooled baskets
    counts_dict: the X most frequent items and their frequency in the 200 ms baskets
    """
    items = list(counts_dict.keys())
    one_hot_counts = one_hot_df.sum(axis=0, skipna=True)
    _ = one_hot_counts.to_dict()

    pooled_counts = {item: freq for item, freq in _.items() if item in items}
    return pooled_counts

def get_lch_order(items_dict, synset_mapping):
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
    lch_df = pd.DataFrame(data=lch_matrix, index=items, columns=items)

    Z = linkage(lch_matrix, 'ward')
    den = dendrogram(Z,
        orientation='top',
        labels=items,
        leaf_font_size=9,
        distance_sort='ascending',
        show_leaf_counts=True)
    plt.savefig('./results/figures/lch_dendrogram.pdf')
    orderedNames = den['ivl']

    return orderedNames, lch_df

def get_w2v_order(freq_items_dict):
    w2v = []
    items = list(freq_items_dict.keys())
    for item in items:
        w2v.append(model[item])

    w2v = np.array(w2v,dtype = float)
    rdm_w2v = squareform(pdist(w2v,metric='correlation'))
    w2v_df = pd.DataFrame(data=rdm_w2v, index=items, columns=items) 
    Z = linkage(squareform(rdm_w2v), 'ward')

    den = dendrogram(Z,
        orientation='top',
        labels=list(freq_items_dict.keys()),
        leaf_font_size=9,
        distance_sort='ascending',
        show_leaf_counts=True)
    plt.savefig('./results/figures/w2v_dendrogram.pdf')
    orderedNames = den['ivl']

    return w2v_df, orderedNames

def create_leverage_matrix(itemsets, counts_dict, mapping):
    """
    itemsets: list of baskets, either pooled or not
    counts_dict: a dictionary of each of the most frequent items and their frequency 
        --> if pooled then this is the dictionary returned from pooled_frequent_items()
    mapping: the integer value mapping for each of the frequent items
    """
    items = list(counts_dict.keys())
    encoded_items = [mapping[k] for k in items]

    #first create the X * 1 probability matrix
    single_probs = {k: v/len(itemsets) for k,v in counts_dict.items()}
    encoded = {mapping.get(k, k): v for k,v in single_probs.items()}
    single_probs_df = pd.DataFrame.from_dict(encoded, orient='index')
    single_probs_df = single_probs_df.reindex(encoded_items)
    #single_probs_array = single_probs_df.values

    #clean itemsets for efficiency
    clipped_itemsets = [[i for i in basket if i in items] for basket in itemsets]
    encoded_itemsets = [[mapping[k] for k in basket] for basket in clipped_itemsets]

    #now create the X*X conditional probability matrix
    pair_counts = np.zeros((150,150))

    for basket in encoded_itemsets:
        for idx, x in enumerate(basket[:-1]):
            for y in basket[idx+1 :]:
                pair_counts[x,y] += 1

    pair_probs = pair_counts/len(itemsets)
    pair_probs = pair_probs + np.transpose(pair_probs)
    pair_probs_df = pd.DataFrame(data=pair_probs, index=encoded_items, columns=encoded_items)

    lev_df = pair_probs_df - (single_probs_df.dot(single_probs_df.T))
    lev_df.index = items
    lev_df.columns = items

    return lev_df

def plot_matrix(df, order, outpath):
    df = df.reindex(order, columns=order)

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
    
    lch_order, lch_df = get_lch_order(single_counts, synset_mapping=item_synsets)
    w2v_df, w2v_order = get_w2v_order(single_counts)
    
    plot_matrix(lch_df, lch_order, outpath='./results/figures/lch_matrix.pdf')
    plot_matrix(w2v_df, lch_order, outpath='./results/figures/w2v_matrix.pdf')
    plot_matrix(w2v_df, w2v_order, outpath='./results/figures/w2v_matrix_orderedw2v.pdf')

    lev_df = create_leverage_matrix(itemsets, single_counts, mapping)
    plot_matrix(lev_df, lch_order, outpath='./results/figures/leverage_matrix_200.pdf')

    #pool baskets into latency 800 ms, 700 ms, 2000 ms)
    pooled = []
    for pool in range(4,11,3):
        pooled_itemsets = pool_baskets(itemsets, pool)
        pooled.append(pooled_itemsets)
        print('For the {} group there are {} number of baskets'.format(pool,len(pooled_itemsets)))

    for i in range(len(pooled)):
        one_hot_items = one_hot(pooled[i])
        pool_count = pooled_frequent_items(one_hot_items, single_counts)
        lev_df = create_leverage_matrix(pooled[i], pool_count, mapping)
        plot_matrix(lev_df, lch_order, outpath='./results/figures/leverage_matrix_{}.pdf'.format(i))

    