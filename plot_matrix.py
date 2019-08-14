import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import seaborn as sns

#Create the dictionary for all possible pairwise comparisons
itemsets = pd.read_csv('./results/frequent_itemsets/frequent_itemsets_one_indv.csv', sep=',')
itemsets = itemsets[:181]
items = itemsets['itemsets'].tolist()
items = [i[12:-3] for i in items]

series_1 = []
for x in items:
    for i in range(181):
        series_1.append(x)

series_2 = []
for i in range(0, len(items)):
    series_2.extend(items)

all_combos_list = list(zip(series_1, series_2))
all_combos_dict = {k: 0 for k in all_combos_list}

#Create the dictionary for actual leverage values
lev = pd.read_csv('./results/association_rules/association_rules_one_indv.csv', sep=',')
leverage = lev[['antecedents','consequents','leverage']]

ant = leverage['antecedents'].tolist()
ant = [i[12:-3] for i in ant]
cons = leverage['consequents'].tolist()
cons = [i[12:-3] for i in cons]
ant_con_list = list(zip(ant, cons))

lev = leverage['leverage'].tolist()
lev_list = [0 if i<0 else i for i in lev]

lev_dict = {k:v for k,v in zip(ant_con_list,lev)} #actual data points

#Compare and update leverage values
symmetric_pairs_list = [(x,y) for x, y in all_combos_dict.keys() if x == y]
symmetric_pairs_dict = {k:1 for k in symmetric_pairs_list}

lev_dict.update(symmetric_pairs_dict) #leverage data and diagonal ones

not_in_rekog_data = {key:0 for key in all_combos_dict if not key in lev_dict} 

lev_dict.update(not_in_rekog_data) #lev_dict is now the correct values for th full 181*181 matrix
lev_dict = collections.OrderedDict(lev_dict)
for key in all_combos_list:
        lev_dict[key] = lev_dict.pop(key)

lev_df = pd.DataFrame()
lev_df['Antecedent'] = series_1
lev_df['Consequent'] = series_2 
lev_df['Leverage'] = list(lev_dict.values())

lev_matrix = lev_df.pivot('Antecedent', 'Consequent', 'Leverage')
ax = sns.heatmap(lev_matrix)
plt.show()

'''
matrix = np.eye(183,183)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(matrix, interpolation='nearest', cmap=cm.Greys_r)
plt.show()