import pandas as pd

ser2 =pd.Series([1,0,2,2,0])
ser3 = pd.Series([1,1,0,0,3])
ser4 = pd.Series([0,0,1,1,0])
ser5 = pd.Series([2,3,0,1,0])
ser6 = pd.Series([0,3,1,0,1])
df = pd.concat([ser2, ser3,ser4,ser5,ser6], axis=1)

def make_symm(df, idx1, idx2):
    a=idx1
    b=idx2
    if df.iloc[a][b] != df.iloc[b][a] and df.iloc[a][b]!=0:
        df.iloc[b][a] = df.iloc[a][b]
    if df.iloc[b][a] != df.iloc[a][b] and df.iloc[b][a]!=0:
        df.iloc[a][b] = df.iloc[b][a]
    return df

for i in range(0,len(df)+1):
    for j in range(0,len(df)+1):
        make_symm(lev_df,i,j)


    '''
    print('begin symmetry')
    for k1, v1 in leverage_dict.items():
        for k2, v2 in leverage_dict.items():
            if k1[0] == k2[1] and k1[1] == k2[0] and not v1 == 0:
                leverage_dict[k2] = v1
    print('complete symmetry')
    '''
    

import copy
import random

a = ['a', 'b', 'c']
b = ['d', 'e', 'f', 'g']
c = ['x', 'y', 'z']
d = ['l', 'm', 'n', 'o', 'p']

pooled = []
pooled.append(a)
pooled.append(b)
pooled.append(c)
pooled.append(d)

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
    count = 0
    while count < 1000000:
        idx = range(len(_))
        i1, i2 = random.sample(idx, 2)
        _[i1], _[i2] = _[i2], _[i1]
        count += 1
    return _

items_shuffled = shuffle_items(pooled)
basket_shuffled = shuffle_baskets(pooled)

print(pooled)
print(items_shuffled)
print(basket_shuffled)

"""From here is for LCH ordering """
from nltk.corpus import wordnet as wn
from scipy.cluster.hierarchy import dendrogram, linkage

