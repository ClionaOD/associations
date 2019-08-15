import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

def create_all_pairwise(items_path):
        items = pd.read_csv(items_path, sep=',')
        items = items['itemsets'].tolist()
        items = [i[12:-3] for i in items]
        items = [i for i in items if not "', '" in i]       

        series_1 = []
        for x in items:
                for i in range(len(items)):
                        series_1.append(x)

        series_2 = []
        for i in range(0, len(items)):
                series_2.extend(items)

        all_combos_list = list(zip(series_1, series_2))
        all_combos_dict = {k: 0 for k in all_combos_list}  

        return all_combos_list, all_combos_dict, series_1, series_2

def create_lev_dict(rules_path):
        lev = pd.read_csv(rules_path, sep=',')

        ant = lev['antecedents'].tolist()
        ant = [i[12:-3] for i in ant]
        cons = lev['consequents'].tolist()
        cons = [i[12:-3] for i in cons]
        ant_con_list = list(zip(ant, cons))

        lev = lev['leverage'].tolist()
        lev_list = [0 if i<0 else i for i in lev]
        lev_dict = {k:v for k,v in zip(ant_con_list,lev_list)} 

        return lev_dict

def create_alphabetical_matrix(pairwise_pool, leverage_pool):
        pairwise_dict = pairwise_pool[1]
        pairwise_list = pairwise_pool[0]
        lev_dict = leverage_pool

        symmetric_pairs_list = [(x,y) for x, y in pairwise_dict.keys() if x == y]
        symmetric_pairs_dict = {k:1 for k in symmetric_pairs_list}
        lev_dict.update(symmetric_pairs_dict) 

        not_in_rekog_data = {key:0 for key in pairwise_dict if not key in lev_dict} 
        lev_dict.update(not_in_rekog_data) 
        lev_dict = collections.OrderedDict(lev_dict)
        for key in pairwise_list:
                lev_dict[key] = lev_dict.pop(key)

        lev_df = pd.DataFrame()
        lev_df['Antecedent'] = pairwise_pool[2]
        lev_df['Consequent'] = pairwise_pool[3]
        lev_df['Leverage'] = list(lev_dict.values())

        lev_matrix = lev_df.pivot('Antecedent', 'Consequent', 'Leverage')
        ax = sns.heatmap(lev_matrix, center=.05, vmin=0, vmax=0.1)
        plt.savefig('./results/figures/association_matrix_test.png')
        plt.close()

        return lev_matrix

def hierarchical_clustering(matrix, pairwise_pool):
        y = matrix.to_numpy()
        linked = linkage(y, method='ward')
        label_list = pairwise_pool[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        dend = dendrogram(linked, color_threshold=250, orientation='top', labels=label_list, distance_sort='descending', show_leaf_counts=True, ax=ax)
        plt.title('Dendrogram')
        ax.tick_params(axis='x', which='major', labelsize=3)
        ax.tick_params(axis='y', which='major', labelsize=5)
        plt.tight_layout()
        plt.savefig('./results/figures/dendrogram_leverage_matrix.png')
        plt.close()

        return dend

if __name__ == "__main__":
        items_path = './results/frequent_itemsets/frequent_itemsets_%d_indv.csv' % 1
        rules_path = './results/association_rules/association_rules_%d_indv.csv' % 1
        
        pairwise_pool_one = create_all_pairwise(items_path)
        levs_pool_one = create_lev_dict(rules_path)

        alphabetical_matrix = create_alphabetical_matrix(pairwise_pool_one, levs_pool_one)

        R = hierarchical_clustering(alphabetical_matrix, pairwise_pool_one)

'''
rdm = []
rdm=distance.squareform(distance.pdist(lev_matrix,metric='correlation'))

multi = MDS(n_components=3, dissimilarity='precomputed', random_state=1)
out = multi.fit_transform(rdm)

plt.scatter(out[:, 0], out[:, 1])
plt.axis('equal')
plt.gcf().set_size_inches((20, 20)) 
plt.show()
'''
