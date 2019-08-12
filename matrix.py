import pandas as pd

itemsets = pd.read_csv('./results/frequent_itemsets/frequent_itemsets_one_indv.csv', sep=',')
itemsets = itemsets[:181]
items = itemsets['itemsets'].tolist()
itemsets['itemsets'] = [i[12:-3] for i in items]