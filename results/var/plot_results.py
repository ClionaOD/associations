import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.loadtxt('./results/var/pvalues/pval_array_0.txt')

nlags = 4
neqs = 150

allpvalues_y1 = np.zeros((neqs,3))
for lag in range(nlags):
    allpvalues_y1[lag,:]=x[1+lag*neqs:4+lag*neqs,0] 

allpvalues_y2 = np.zeros((neqs,3))
for lag in range(nlags):
    allpvalues_y2[lag,:]=x[1+lag*neqs:4+lag*neqs,1]  

allpvalues_y3 = np.zeros((neqs,3))
for lag in range(nlags):
    allpvalues_y3[lag,:]=x[1+lag*neqs:4+lag*neqs,2] 