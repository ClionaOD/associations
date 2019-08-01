import pickle
import glob
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def create_baskets(dict1):

    uniquetimestamps = set([x['Timestamp'] for x in dict1['alllabels']])
    basket = { i: [] for i in uniquetimestamps}
    return basket

def fill_baskets(dict1, dict2):
    #dict1 is the empty baskets for each movie (i.e. basket from create_baskets)
    #dict2 is the same as create_baskets' arg, each movie's labels.
    for vals in dict2['alllabels']:
        dict1[vals['Timestamp']].extend([vals['Label']['Name']])
    
    return dict1

def pool_baskets(inlist, multiply_frames=1):
    #inlist is the itemsets list of lists
    #multiple_frames is what to multiply 200 ms by to get desired pooling e.g. multiply_frames=10 for 2 second baskets
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
        
    files = glob.glob('/home/CUSACKLAB/clionaodoherty/associations/data/*.pickle')
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

    itemsets = []
    for movie in movies:
        baskets = fill_baskets(create_baskets(movie), movie)
        itemsets.extend(baskets.values())

    for pool in range(1,11,3):
        pooled_itemsets = pool_baskets(itemsets, pool)
        
        te = TransactionEncoder()
        te_ary = te.fit(itemsets).transform(pooled_itemsets, sparse=True)
        df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

        frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True, verbose=1, max_len=2)
        frequent_itemsets.to_csv(r'/home/CUSACKLAB/clionaodoherty/associations/results/frequent_itemsets_%.0f.csv' %pool, sep=',', index=False)

        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
        rules = rules[rules.confidence != 1]
        rules.to_csv(r'/home/CUSACKLAB/clionaodoherty/associations/results/association_rules_%.0f.csv' %pool, sep=',', index=False)
