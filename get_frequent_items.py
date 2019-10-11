import pickle
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def one_hot(lst): 
    """one hot encode lst, a list of lists"""
    te = TransactionEncoder()
    te_ary = te.fit(lst).transform(lst, sparse=True)
    df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)
    return df

with open('itemsets.pickle', 'rb') as f:
    itemsets = pickle.load(f)

one_hot_items = one_hot(itemsets)

single_counts = one_hot_items.sum(axis=0, skipna=True) #this is a bit slow but works
top_150 = single_counts.nlargest(150, keep='all')

counts_one = top_150.to_dict()
probs_one = {k: v/len(itemsets) for k,v in counts_one.items()}

item_to_number = {k: v for v, k in enumerate(sorted(set(probs_one.keys())))}
encode_probs_one = {item_to_number.get(k, k): v for k, v in probs_one.items()}

prob_one_df = pd.DataFrame.from_dict(encode_probs_one, orient='index')
prob_one_array = df.to_numpy

pair_counts = np.zeros((150,150))

for basket in itemsets:
    for idx, x in enumerate(itemsets[0][:-1]):
        print(idx,x)

