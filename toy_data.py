import pickle
import json
import get_frequent_items as freq

with open('itemsets.pickle', 'rb') as f:
    itemsets = pickle.load(f)

toy_itemsets = itemsets[:20]

with open('./scratch/toy_itemsets.json', 'w') as write_file:
    json.dump(toy_itemsets, write_file)

X = 10

one_hot = freq.one_hot(toy_itemsets)
counts, maps = freq.most_frequent_items(one_hot, X)

with open('./scratch/toy_counts.json', 'w') as f:
    json.dump(counts, f)

with open('./scratch/toy_maps.json', 'w') as f:
    json.dump(maps, f)

toy_order = list(counts.keys())

with open('./scratch/toy_order.json', 'w') as f:
    json.dump(toy_order, f)