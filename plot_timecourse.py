import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

threshold = 0.001

with open('./results/ridge_regression/divided_offDiagonals_linspace.pickle','rb') as f:
    diags=pickle.load(f)

meanVals = np.mean(diags,axis=2)
remove = np.where(meanVals[:,0] < threshold)[0]
vals = np.delete(meanVals,remove,0)

fig,ax = plt.subplots(figsize=[25,13])
ax.plot(vals.T)
ax.set_title('Timecourse of the off diagonal values (average over 16 divs)')
ax.set_xlabel('lags (regression done on linearly spaced lags)')
ax.set_xticks(range(40))
ax.set_xticklabels(list(np.linspace(1,36000,num=40, dtype=int)))
ax.set_ylabel('pairwise coefficients of the off diagonal (thresholded at B > {})'.format(threshold))
plt.savefig('./results/ridge_regression/figs/timecourses/meanOffDiag_linear_timecourse_(threshold {}).pdf'.format(threshold))