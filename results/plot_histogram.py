import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_itemsets(itemspath):
    itemsets = pd.read_csv(itemspath, sep=',')
    itemsets = itemsets['itemsets'].tolist()
    itemsets = [i[12:-3] for i in itemsets]
    itemsets = [i for i in itemsets if not "', '" in i]
    return itemsets

one = read_itemsets('./results/frequent_itemsets/itemsets_1.csv')
four = read_itemsets('./results/frequent_itemsets/itemsets_4.csv')
seven = read_itemsets('./results/frequent_itemsets/itemsets_7.csv')
ten = read_itemsets('./results/frequent_itemsets/itemsets_10.csv')

bask_one = read_itemsets('./results/frequent_itemsets/basket_shuffle_itemsets_1.csv')
bask_four = read_itemsets('./results/frequent_itemsets/basket_shuffle_itemsets_4.csv')
bask_seven = read_itemsets('./results/frequent_itemsets/basket_shuffle_itemsets_7.csv')
bask_ten = read_itemsets('./results/frequent_itemsets/basket_shuffle_itemsets_10.csv')

items_one = read_itemsets('./results/frequent_itemsets/item_shuffle_itemsets_1.csv')
items_four = read_itemsets('./results/frequent_itemsets/item_shuffle_itemsets_4.csv')
items_seven = read_itemsets('./results/frequent_itemsets/item_shuffle_itemsets_7.csv')
items_ten = read_itemsets('./results/frequent_itemsets/item_shuffle_itemsets_10.csv')


a = [i for i in four if not i in items_four]
b = [i for i in four if not i in bask_four]
c = [i for i in items_four if not i in four]
d = [i for i in bask_four if not i in four]

#items_four/bask_four = four + Cafeteria Window 

#one is a subset of all other three
#four = one + Bed, Grass, Meal, Military, Tuxedo
#seven = four + computer, flooring
#ten = seven + cafeteria, window - military



"""
Plot the actual distribution for leverage first
"""
rules_one = pd.read_csv('./results/association_rules/association_rules_1.csv', sep=',')
rules_four = pd.read_csv('./results/association_rules/association_rules_4.csv', sep=',')
rules_seven = pd.read_csv('./results/association_rules/association_rules_7.csv', sep=',')
rules_ten = pd.read_csv('./results/association_rules/association_rules_10.csv', sep=',')

lev_one = rules_one[['antecedents','consequents','leverage']]
lev_four = rules_four[['antecedents','consequents','leverage']]
lev_seven = rules_seven[['antecedents','consequents','leverage']]
lev_ten = rules_ten[['antecedents','consequents','leverage']]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax[0,0].hist(rules_one['leverage'], bins=100)
ax[0,0].set_xlim([-0.05, 0.15])
ax[0,1].hist(rules_four['leverage'], bins=100)
ax[0,1].set_xlim([-0.05, 0.15])
ax[1,0].hist(rules_seven['leverage'], bins=100)
ax[1,0].set_xlim([-0.05, 0.15])
ax[1,1].hist(rules_ten['leverage'], bins=100)
ax[1,1].set_xlim([-0.05, 0.15])
ax[0,0].set_title('200 ms, 0.021 support')
ax[0,1].set_title('800 ms, 0.075 support')
ax[1,0].set_title('1400 ms, 0.120 support')
ax[1,1].set_title('2000 ms, 0.163 support')

plt.suptitle('Leverage Distribution for Real Data')
plt.savefig('./results/figures/v3/leverage_distribution.pdf')
plt.close()

"""
Now the distribution having shuffled all items between baskets
"""
rules_one = pd.read_csv('./results/association_rules/item_shuffle_association_rules_1.csv', sep=',')
rules_four = pd.read_csv('./results/association_rules/item_shuffle_association_rules_4.csv', sep=',')
rules_seven = pd.read_csv('./results/association_rules/item_shuffle_association_rules_7.csv', sep=',')
rules_ten = pd.read_csv('./results/association_rules/item_shuffle_association_rules_10.csv', sep=',')

lev_one = rules_one[['antecedents','consequents','leverage']]
lev_four = rules_four[['antecedents','consequents','leverage']]
lev_seven = rules_seven[['antecedents','consequents','leverage']]
lev_ten = rules_ten[['antecedents','consequents','leverage']]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax[0,0].hist(rules_one['leverage'], bins=100)
ax[0,0].set_xlim([-0.05, 0.15])
ax[0,1].hist(rules_four['leverage'], bins=100)
ax[0,1].set_xlim([-0.05, 0.15])
ax[1,0].hist(rules_seven['leverage'], bins=100)
ax[1,0].set_xlim([-0.05, 0.15])
ax[1,1].hist(rules_ten['leverage'], bins=100)
ax[1,1].set_xlim([-0.05, 0.15])
ax[0,0].set_title('200 ms, 0.021 support')
ax[0,1].set_title('800 ms, 0.075 support')
ax[1,0].set_title('1400 ms, 0.120 support')
ax[1,1].set_title('2000 ms, 0.163 support')

plt.suptitle('Leverage Distribution for Shuffled Data (items shuffled)')
plt.savefig('./results/figures/v3/item_shuffle_leverage_distribution.pdf')
plt.close()

"""
Finally, randomly shuffling the order of baskets
"""
rules_one = pd.read_csv('./results/association_rules/basket_shuffle_association_rules_1.csv', sep=',')
rules_four = pd.read_csv('./results/association_rules/basket_shuffle_association_rules_4.csv', sep=',')
rules_seven = pd.read_csv('./results/association_rules/basket_shuffle_association_rules_7.csv', sep=',')
rules_ten = pd.read_csv('./results/association_rules/basket_shuffle_association_rules_10.csv', sep=',')

lev_one = rules_one[['antecedents','consequents','leverage']]
lev_four = rules_four[['antecedents','consequents','leverage']]
lev_seven = rules_seven[['antecedents','consequents','leverage']]
lev_ten = rules_ten[['antecedents','consequents','leverage']]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax[0,0].hist(rules_one['leverage'], bins=100)
ax[0,0].set_xlim([-0.05, 0.15])
ax[0,1].hist(rules_four['leverage'], bins=100)
ax[0,1].set_xlim([-0.05, 0.15])
ax[1,0].hist(rules_seven['leverage'], bins=100)
ax[1,0].set_xlim([-0.05, 0.15])
ax[1,1].hist(rules_ten['leverage'], bins=100)
ax[1,1].set_xlim([-0.05, 0.15])
ax[0,0].set_title('200 ms, 0.021 support')
ax[0,1].set_title('800 ms, 0.075 support')
ax[1,0].set_title('1400 ms, 0.120 support')
ax[1,1].set_title('2000 ms, 0.163 support')

plt.suptitle('Leverage Distribution for Shuffled Data (baskets shuffled)')
plt.savefig('./results/figures/v3/basket_shuffle/basket_shuffle_leverage_distribution.pdf')
plt.close()