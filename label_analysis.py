import pickle
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

file_Name = 'Peppa-pig-edit.pickle'
fileObject = open(file_Name, 'rb')
movie_one_labels = pickle.load(fileObject)

del movie_one_labels['compmsg']
del movie_one_labels['deltat']
del movie_one_labels['vid']

timed_labels = {}
for i in movie_one_labels.values():
    for time in i:
        if not time['Timestamp'] in timed_labels:
            timed_labels[time['Timestamp']] = [] #at this point we have dict with k timestamp and v empty list

for x in timed_labels:
    for i in movie_one_labels.values():
        for time in i:
            if time['Timestamp'] == x:
                timed_labels[time['Timestamp']] += time['Label'].values() #this has v as a list with Name and Confidence

for x in timed_labels.values():
    del x[1::2]
    del x[1::2] #removes confidence values

itemsets = list(timed_labels.values())
sets = []
for i in itemsets:
    sets.append(i[0::4])

df = pd.DataFrame(sets)
df.to_csv(r'/home/clionaodoherty/Desktop/Association_Movies/peppa-tags.csv', sep=',', index=False)

lst = itemsets[0]

itemsets.to_csv(r'/home/clionaodoherty/Desktop/Association_Movies/jungle_labels.csv', sep=',', index=False)
for x in itemsets:
    for i in x:
        del x[1::2]

#timed_labels is now a dict with k as timestamps and v as a list of the labels at this time

itemsets = list(timed_labels.values())
te = TransactionEncoder()
te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True, verbose=1) #support is set to 2%, approx 8 occurences of both A and B over total number (4075)
frequent_itemsets.to_csv(r'/home/clionaodoherty/Desktop/Association_Movies/jungle_apriori_results.csv', sep=',', index=False)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=.9)
rules.to_csv(r'/home/clionaodoherty/Desktop/Association_Movies/jungle_association_rules.csv', sep=',', index=False)