import numpy as np
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from scipy import stats

def hierarchical_clustering(matrix, label_list, outpath=None):
    fig,ax = plt.subplots(figsize=(10,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), 
        ax=ax, 
        labels=label_list, 
        orientation='left'
    )
    ax.tick_params(axis='x', labelsize=4)
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order

if __name__ == "__main__":
    
    loadPath = './results/coefficients'
    orderPath = './freq_order.pickle'
    savePath = './results/figs/matrices/'

    with open('{}/all-betas.pickle'.format(loadPath), 'rb') as f:
        all_betas = pickle.load(f)

    with open(orderPath,'rb') as f:
        order = pickle.load(f)

    mins = [3,15,40,50,75,100,120] #mins chosen to analyse
    matrix_lags = [1,5,13,17,24,33,-1] #indices of the lags chosen from R2 graph from temporal_regression.py

    mean_coefs = np.mean(all_betas, axis=3)

    #get hierarchical clustering order of the first regression 
    clusterOrder = hierarchical_clustering(mean_coefs[:,:,matrix_lags[0]], order, outpath='./results/figs/dendrogram.pdf')

    coefMax = 0.015
    coefMin = -0.015

    for idx, lag in enumerate(matrix_lags):
        plotMatrix = pd.DataFrame(mean_coefs[:,:,lag], index=order, columns=order)
        plotMatrix = plotMatrix.reindex(index=clusterOrder, columns=clusterOrder)

        fig,ax = plt.subplots(figsize=[20,15])
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(plotMatrix,
            ax=ax, 
            cmap=cmap, 
            vmin=coefMin, 
            vmax=coefMax,
            center=0.0)
        ax.axes.set_title('Mean coefficients {} min lag'.format(mins[idx]), fontsize=30)
        ax.tick_params(labelsize=7)
        #plt.show()
        plt.savefig('{}/meanCoefs_{}mins.pdf'.format(savePath, mins[idx]))
