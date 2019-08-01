import pandas as pd

one = pd.read_csv('./results/association_rules_pooled_1.csv', sep=',')
four = pd.read_csv('./results/association_rules_pooled_4.csv', sep=',')
seven = pd.read_csv('./results/association_rules_pooled_7.csv', sep=',')
ten = pd.read_csv('./results/association_rules_pooled_10.csv', sep=',')

leverage_one = one['leverage']
leverage_four = four['leverage']
leverage_seven = seven['leverage']
leverage_ten = ten['leverage']

print(leverage_one.describe())
print(leverage_four.describe())
print(leverage_seven.describe())
print(leverage_ten.describe())

independent = ten[ten.lift == 1]

