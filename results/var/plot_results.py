import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

x = np.loadtxt('./results/var/pvalues/pval_array_1.txt')

nlags = 4
neqs = 150

'''
allpvalues_y1 = np.zeros((neqs,3))
for lag in range(nlags):
    allpvalues_y1[lag,:]=x[1+lag*neqs:4+lag*neqs,0] 

allpvalues_y2 = np.zeros((neqs,3))
for lag in range(nlags):
    allpvalues_y2[lag,:]=x[1+lag*neqs:4+lag*neqs,1]  

allpvalues_y3 = np.zeros((neqs,3))
for lag in range(nlags):
    allpvalues_y3[lag,:]=x[1+lag*neqs:4+lag*neqs,2] 
'''

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





        