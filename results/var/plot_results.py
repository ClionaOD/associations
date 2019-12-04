import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

x = np.loadtxt('./results/var/pvalues/pval_array_0.txt')

nlags = 4
neqs = 150

eqs = [['L1.'], ['L2.'], ['L3.'], ['L4.']]
yeqs = []
for lst in eqs:
    for i in range(neqs-1):
        lst.append(lst[0])

    
    for i in range(neqs):
        y = 'y{}'.format(i+1)
        yeqs.append(y)
        lst[i] = lst[i] + y 
        

yeqs = yeqs[:150]

eqs = [eq for lst in eqs for eq in lst]
eqs = ['const'] + eqs # a list of all possible lags and equations

pvals = pd.DataFrame(x, index = eqs, columns = yeqs)
sns.heatmap(pvals, cmap='coolwarm')
plt.show()

coef = np.loadtxt('./results/var/coefs/coef_array_0_3.txt')
sns.heatmap(coef, cmap='coolwarm')
plt.show()