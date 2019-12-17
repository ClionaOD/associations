import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('./results/ridge_regression/diagonals_linspace.pickle','rb') as f:
    diags=pickle.load(f)

vals = diags.values

fig,ax = plt.subplots(figsize=[25,13])
ax.plot(vals.T)
ax.set_title('Timecourse of diagonal (entire dataset)')
ax.set_xlabel('lags (regression done on linearly spaced lags)')
ax.set_xticks(range(40))
ax.set_xticklabels(list(diags.columns))
ax.set_ylabel('pairwise coefficients of diagonal')
plt.savefig('./results/ridge_regression/linear_timecourse_plt.pdf')