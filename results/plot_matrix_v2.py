import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

def create_leverage_dict(itemspath, rulespath):
    itemsets = pd.read_csv(itemspath, sep=',')
    itemsets = itemsets['itemsets'].tolist()
    itemsets = [i[12:-3] for i in itemsets]
    itemsets = [i for i in itemsets if not "', '" in i]  #itemsets is a list of n length for all single frequent itemsets

    antecedents = []
    for x in itemsets:
        for i in range(len(itemsets)):
            antecedents.append(x)

    consequents = []
    for i in range(0, len(itemsets)):
        consequents.extend(itemsets)

    compare_tuples = list(zip(antecedents, consequents))

    leverages = pd.read_csv(rulespath, sep=',') 
    ants = leverages['antecedents'].tolist()
    ants = [i[12:-3] for i in ants]
    cons = leverages['consequents'].tolist()
    cons = [i[12:-3] for i in cons]
    lev = leverages['leverage'].tolist()
    
    leverage_tuples = list(zip(ants, cons))
    leverage_values = [0 if i<0 else i for i in lev]

    compare_dict = {k:0 for k in compare_tuples} 
    leverage_dict = {k:v for k,v in zip(leverage_tuples, leverage_values)} 

    #update the leverage_dict to include values for item*item as well as items not in leverages
    diagonals = [(x,y) for x, y in compare_dict.keys() if x == y]
    diagonals = {k:1 for k in diagonals}
    leverage_dict.update(diagonals) 

    absences = {k:0 for k in compare_dict if not k in leverage_dict} 
    leverage_dict.update(absences) 

    for k1, v1 in leverage_dict.items():
        for k2, v2 in leverage_dict.items():
            if k1[0] == k2[1] and k1[1] == k2[0] and not v1 == 0:
                leverage_dict[k2] = v1

    return leverage_dict, antecedents, consequents, compare_tuples, itemsets

def reorder_od(dict1,order):
    new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
    new_od.update(dict1)
    return new_od

def create_matrix(lev_dict, series1, series2, reind_order, outpath):
    df = pd.DataFrame()
    df['Antecedents'] = series1
    df['Consequents'] = series2
    df['Leverage'] = list(lev_dict.values())

    df = df.pivot('Antecedents', 'Consequents', 'Leverage')
    df = df.reindex(reind_order, columns=reind_order)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df, center=.05, vmin=0, vmax=0.1, ax=ax)
    plt.savefig(outpath)
    plt.close()

    leverage_array = df.to_numpy()

    return df, leverage_array

def hierarchical_clustering(matrix, label_list, outpath):
    fig,ax = plt.subplots(figsize=(10,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), ax=ax, labels=label_list)
    ax.tick_params(axis='x', labelsize=4)
    plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order

def mds(matrix,outpath):
    distance = squareform(pdist(matrix, metric='correlation'))

    mds = MDS(n_components=2, dissimilarity='precomputed')
    pos = mds.fit(distance).embedding_

    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])
    plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', lw=0, label='MDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)

    #plt.show()
    plt.savefig(outpath)
    plt.close()

if __name__ == "__main__":
    for pool in range(1,11,3):
        itemspath = './results/frequent_itemsets/frequent_itemsets_%d_indv.csv' %pool
        rulespath = './results/association_rules/association_rules_%d_indv.csv' %pool

        leverage_dict, antecedents, consequents, order, labels = create_leverage_dict(itemspath=itemspath, rulespath=rulespath)

        alphab_dict = reorder_od(leverage_dict, order)

        alphab_outpath = './results/figures/v2/association_matrix_alphabetical_%d.pdf' %pool
        lev_df, lev_array = create_matrix(lev_dict=alphab_dict, series1=antecedents, series2=consequents, reind_order=labels, outpath=alphab_outpath) 

        dendro_outpath = './results/figures/v2/dendrogram_%d.pdf' %pool
        clusters = hierarchical_clustering(matrix=lev_array, label_list=labels, outpath=dendro_outpath)

        cluster_ants = []
        for x in clusters:
            for i in range(len(clusters)):
                cluster_ants.append(x)

        cluster_cons = []
        for i in range(0, len(clusters)):
            cluster_cons.extend(clusters)

        cluster_tuples = list(zip(cluster_ants, cluster_cons))

        cluster_dict = reorder_od(leverage_dict, cluster_tuples)
        cluster_outpath = './results/figures/v2/association_matrix_clustered_%d.pdf' %pool
        cluster_df, cluster_array = create_matrix(lev_dict=cluster_dict, series1=cluster_ants, series2=cluster_cons, reind_order=clusters, outpath=cluster_outpath)
