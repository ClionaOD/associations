import pickle
import glob
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

files = glob.glob('/home/clionaodoherty/Desktop/Association_Movies/labels/*.pickle')
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

print(len(movies[0].values()))
#Movies is now a list of dicts.

def create_baskets(dict):
    basket = {}
    for i in dict.values():
        for time in i:
            if not time['Timestamp'] in basket:
                basket[time['Timestamp']] = []
    return basket

def fill_baskets(dict1, dict2):
    #dict1 is the empty baskets for each movie (from create_baskets)
    #dict2 is the same as create_baskets arg, each movie's labels.
    for x in dict1:
        for i in dict2.values():
            for time in i:
                if time['Timestamp'] == x:
                    dict1[time['Timestamp']] += time['Label'].values()
    
    for x in dict1.values():
        del x[1::2]
        del x[1::2]
    
    return dict1

itemsets = []
for movie in movies:
    baskets = fill_baskets(create_baskets(movies[0]), movies[0])
    item_dicts.extend(baskets.values())

def pool_baskets(dict):
    inlist = list(dict.values())
    rest = ['Logo', 'Text', 'Trademark', 'Word']
    inlist = [x for x in inlist if x != rest]

    outlist = []
    startpoint = 0

    for j in range(0,len(inlist)//10):
        endpoint = startpoint + 10
        outelement = []
        list_temp = inlist[startpoint:endpoint]
        for i in range(0,10):
            outelement.extend(list_temp[i])
        outlist.append(outelement)
        startpoint = startpoint + 10

    if len(inlist)%10 != 0:
        remainder = len(inlist)%10
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

itemsets.extend(pool_baskets(amelie_baskets))

#compute apriori and association rules
te = TransactionEncoder()
te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True, verbose=1)
frequent_itemsets.to_csv(r'/home/clionaodoherty/Desktop/Association_Movies/amelie_apriori_results.csv', sep=',', index=False)

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules = rules[rules.confidence != 1]
rules.to_csv(r'/home/clionaodoherty/Desktop/Association_Movies/amelie_association_rules.csv', sep=',', index=False)