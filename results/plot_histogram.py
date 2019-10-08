import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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