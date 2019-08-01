import pandas as pd

rules = pd.read_csv('association_rules.csv', sep=',')

antecedents = rules['antecedents'].tolist()
consequents = rules['consequents'].tolist()

antecedents = [i[12:-3] for i in antecedents]
consequents = [i[12:-3] for i in consequents]

pairs = list(zip(antecedents, consequents))
pairs = set(pairs)

a = [(x,y) for x, y in pairs if y  == 'Accessories' ]
[(x,y) for x, y in a if x  == 'Accessories' ]