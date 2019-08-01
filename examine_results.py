import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rules_one = pd.read_csv('./results/association_rules_1.csv', sep=',')
rules = rules_one[['antecedents','consequents','leverage']]

rules_one = pd.read_csv('./results/association_rules_1.csv', sep=',')
rules_four = pd.read_csv('./results/association_rules_4.csv', sep=',')
rules_seven = pd.read_csv('./results/association_rules_7.csv', sep=',')
rules_ten = pd.read_csv('./results/association_rules_10.csv', sep=',')

lev_one = rules_one[['antecedents','consequents','leverage']]
lev_four = rules_four[['antecedents','consequents','leverage']]
lev_seven = rules_seven[['antecedents','consequents','leverage']]
lev_ten = rules_ten[['antecedents','consequents','leverage']]


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax[0,0].hist(rules_one['leverage'], bins=100)
ax[0,1].hist(rules_four['leverage'], bins=100)
ax[1,0].hist(rules_seven['leverage'], bins=100)
ax[1,1].hist(rules_ten['leverage'], bins=100)
ax[0,0].set_title('200 ms')
ax[0,1].set_title('800 ms')
ax[1,0].set_title('1400 ms')
ax[1,1].set_title('2000 ms')

plt.savefig('./results/common_support_5_hists')
