import numpy as np
import pandas as pd
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns

def hierarchical_clustering(matrix, label_list):
    fig,ax = plt.subplots(figsize=(10,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), ax=ax, labels=label_list)
    #ax.tick_params(axis='x', labelsize=4)
    #plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order

if __name__ == "__main__":

    with open('./results/ridge_regression/LCH_allcoefs_extendedlags.pickle', 'rb') as f:
        allcoefs = pickle.load(f)

    with open('lch_order.pickle', 'rb') as f:
        lchOrder = pickle.load(f)

    lags = [1,4500,13500,22500,31500,40500]
    nlags = len(lags)

    coef_tstats=stats.ttest_1samp(allcoefs, 0, axis=3)
    mn_allcoefs=np.mean(allcoefs,axis=3)

    for lag in range(nlags):
        pvals = coef_tstats.pvalue[lag]
        sigPval = np.zeros((150,150))
        sigs = np.where(pvals < 0.01)
        sigs = list(zip(sigs[0], sigs[1]))
        for coord in sigs:
            sigPval[coord[0]][coord[1]] = 1

        coefs = mn_allcoefs[lag]
        threshCoefs = sigPval * coefs
        print('pause')

        """
        df = pd.DataFrame(threshCoefs, index=lchOrder, columns=lchOrder)
        df = df.loc[(df!=0).any(axis=1)]
        
        newItems = list(df.index)
        df = df[newItems]
        
        newOrder = hierarchical_clustering(df,newItems)
        df = df.reindex(newOrder,columns=newOrder)

        fig,ax = plt.subplots(figsize=[20,15])
        sns.heatmap(df,ax=ax, cmap='seismic', xticklabels=newOrder, yticklabels=newOrder, center=0)
        ax.axes.set_title('Mean Betas Lag {} (thresholded by p < 0.01)'.format(lags[lag]), fontsize=35)
        ax.tick_params(labelsize=7)
        plt.savefig('./results/ridge_regression/figs/extendedLags/clustered-thresholded/Lag{}_meanCoefs(selfclustering_thresh).pdf'.format(lags[lag]))
        """


