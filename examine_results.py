import matplotlib.pyplot as plt
import pandas as pd

rules_one = pd.read_csv('./results/association_rules_pooled_1.csv', sep=',')
rules = rules_one[['antecedents','consequents','leverage']]

plt.hist(rules['leverage'], bins=1000)