import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

threshold = 0.015

with open('./results/ridge_regression/offDiagonals_linspace.pickle','rb') as f:
    diags=pickle.load(f)

vals = diags.values
remove = np.where(vals[:,0] < threshold)[0]
vals = np.delete(vals,remove,0)

fig,ax = plt.subplots(figsize=[25,13])
ax.plot(vals.T)
ax.set_title('Timecourse of the off diagonal values (entire dataset)')
ax.set_xlabel('lags (regression done on linearly spaced lags)')
ax.set_xticks(range(40))
ax.set_xticklabels(list(diags.columns))
ax.set_ylabel('pairwise coefficients of the off diagonal (thresholded at B > {})'.format(threshold))
plt.savefig('./results/ridge_regression/linear_timecourse_extendedLags(threshold {}).pdf'.format(threshold))