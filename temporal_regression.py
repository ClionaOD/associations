import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from sklearn.linear_model import Ridge
from scipy import stats

import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

def ridge_regress(X,y):
    clf = Ridge(alpha=1, normalize=True)
    clf.fit(X,y)
    score = clf.score(X,y)
    coefs = clf.coef_
    return coefs, score

def divide_dataset(lst, div):
    """Split the dataset into smaller samples"""
    length = int(len(lst)/div)

    split_lst = []
    mult = 0
    for i in range(div):
        x = lst[length*mult : length*(mult+1)]
        split_lst.append(x)
        mult += 1

    return split_lst

def encode_dataset(lst, items=None):
    """
    Encode the categorical data into a binary array.
    lst: The list of labels (structured as a list of lists).
    items: A list of labels to keep in the encoded array.
    
    Returns array with each row corresponding to a timepoint and each column an item.
    """
    te = TransactionEncoder()
    encoded_lst = te.fit(lst).transform(lst, sparse=False)
    encoded_lst = encoded_lst.astype(int)
    encoded_df = pd.DataFrame(encoded_lst, columns=te.columns_)

    if items:
        if not type(items) == list:
            print('Please provide a list of labels to keep in the DataFrame.')
        else:
            encoded_df = encoded_df[items]
    
    return encoded_df

def most_freq(lst, X=150):
    """Get the top X most frequent items in the dataset"""
    encoded_df = encode_dataset(lst)

    counts = encoded_df.sum(axis=0, skipna=True)
    
    #Choose only those which are in wordnet
    all_items = counts.to_dict()
    wn_items = {item : freq for item, freq in all_items.items() if len(wn.synsets(item, pos='n')) != 0 and item in model.vocab}
    wn_items = pd.DataFrame.from_dict(wn_items, orient='index', columns=['Count'])
    
    top_X = pd.DataFrame(wn_items.nlargest(X, columns=wn_items.columns, keep='all'))

    freq_items = top_X.index.tolist()
    
    return freq_items 

def get_coefs(data, sweeps):
    arr = data.values
    coefs = np.zeros((nitems,nitems,len(sweeps)))
    scores = []
    for lag in range(len(sweeps)):
        y = arr[sweeps[-1]:,:]
        X = arr[sweeps[-1] - sweeps[lag] : -sweeps[lag], :]
        coef, score = ridge_regress(X,y)
        coefs[:,:,lag] = coef
        scores.append(score)

    return coefs, scores

def get_timecourse_coefs(coef_arr, sweeps, nitems=150, divBy=16):
    """
    coef_arr: the 3D array (nitems, nitems, nsweeps, divVy) of coefficients
    sweeps: list of sweeps
    nitems: number of frequent items i.e. len(order)
    """
    diags = np.zeros((nitems, len(sweeps), divBy))
    off_diags = np.zeros(((nitems*nitems)-nitems, len(sweeps), divBy))
    
    for div in range(divBy):
        divDiags = np.zeros((nitems, len(sweeps)))
        divOff_diags = np.zeros(((nitems*nitems)-nitems, len(sweeps)))
        
        for lag in range(len(sweeps)):
            lagCoefs = coef_arr[:,:,lag,div]
            
            d = lagCoefs.diagonal()
            divDiags[:,lag] = d

            offd = lagCoefs[~np.eye(lagCoefs.shape[0],dtype=bool)].reshape(lagCoefs.shape[0],-1)
            offd = offd.reshape(-1)
            divOff_diags[:,lag] = offd
        
        diags[:,:,div] = divDiags
        off_diags[:,:,div] = divOff_diags

    meanDiags = np.mean(diags, axis=2)
    diagTstats = stats.ttest_1samp(diags, 0, axis=2)
    diagPvals = diagTstats.pvalue

    meanOffs = np.mean(off_diags, axis=2)
    off_diagTstats = stats.ttest_1samp(off_diags, 0, axis=2)
    off_diagPvals = off_diagTstats.pvalue
    
    return meanDiags, diagPvals, meanOffs, off_diagPvals

if __name__ == "__main__":
       
    #Set paramaters 
    dataPath = './itemsets.pickle'
    orderPath = './freq_order.pickle'   #Put to None if the frequent items have not yet been computed
    savePath = './results/coefficients'
    loadPath = savePath                 #Set to None if need to calculate all coefficients
    modelPath = '/home/CUSACKLAB/clionaodoherty/GoogleNews-vectors-negative300.bin'

    nitems = 150
    divBy = 16
    
    #Load the data
    with open(dataPath,'rb') as f:
        dataset = pickle.load(f)

    #Get frequent items
    if orderPath:
        with open(orderPath,'rb') as f:
            order = pickle.load(f)
    else:
        #load wordnet model
        nltk.download('wordnet')
        model = KeyedVectors.load_word2vec_format(modelPath, binary=True, limit=500000)

        order = most_freq(dataset, nitems)

        with open('./freq_order.pickle', 'wb') as f:
            pickle.dump(order,f)

    #Divide the dataset to allow for mean calculation
    divDataset = divide_dataset(dataset, divBy)

    #Calculate the range of lags to sweep over
    sweeps = np.linspace(1,36000,num=40, dtype=int)
    sweepMins = []
    for sweep in sweeps:
        sweepMins.append(int(round((sweep*200) / (60*1000))))
    
    #Get coefs over the sweep, separate diagonal and non-diagonal
    if loadPath:
        with open('{}/all-betas.pickle'.format(savePath), 'rb') as f:
            all_betas = pickle.load(f)
        with open('{}/R2_scores.pickle'.format(savePath), 'rb') as f:
            R2_scores = pickle.load(f)
    else:
        all_betas = np.zeros((nitems,nitems,len(sweeps),len(divDataset)))
        R2_scores = []
        
        for i in range(divBy):
            data = encode_dataset(divDataset[i], order)
            coefs, scores = get_coefs(data,sweeps)
            all_betas[:,:,:,i] = coefs
            R2_scores.append(scores)

        with open('{}/all-betas.pickle'.format(savePath), 'wb') as f:
            pickle.dump(all_betas,f)
        
        with open('{}/R2_scores.pickle'.format(savePath), 'wb') as f:
            pickle.dump(R2_scores,f)

    #Average the coef arrays & get stats
    meanCoefs = np.mean(all_betas, axis=3)

    tstatsCoefs = stats.ttest_1samp(all_betas, 0, axis=3)
    pvalsCoefs = tstatsCoefs.pvalue

    #Separate diagonal
    meanDiags, diagPvals, meanOffs, off_diagPvals = get_timecourse_coefs(all_betas,sweeps,nitems=nitems, divBy=divBy)
    
    with open('{}/mean-diagonal-coefs.pickle'.format(savePath),'wb') as f:
        pickle.dump(meanDiags,f)

    with open('{}/mean-off-diagonal-coefs.pickle'.format(savePath),'wb') as f:
        pickle.dump(meanOffs,f)

    #plot the timecourses
    thresh = 0.01
    
    threshold = diagPvals < thresh
    plotDiags = meanDiags.copy()
    plotDiags[~threshold] = float('NaN') 
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(plotDiags.T)
    ax.set_title('Timecourse of the diagonal values (mean)')
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('coefficients of the diagonal (thresholded at p < {})'.format(thresh))
    plt.savefig('./results/figs/timecourses/diagonal-timecourse_(threshold p < {}).pdf'.format(thresh))
    plt.close()

    threshold = off_diagPvals < thresh
    plotOffs = meanOffs.copy()
    plotOffs[~threshold] = float('NaN')
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(plotOffs.T)
    ax.set_title('Timecourse of the off-diagonal values (mean)')
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('coefficients of the off-diagonal (thresholded at p < {})'.format(thresh))
    plt.savefig('./results/figs/timecourses/off-diagonal-timecourse_(threshold p < {}).pdf'.format(thresh))
    plt.close()

    #Plot the R2 values
    R2_arr = np.array(R2_scores)
    means = np.mean(R2_arr, axis=0)
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(means.T)
    ax.set_title('Mean R2 values from 0 - {} mins'.format(sweepMins[-1]))
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('R2')
    plt.savefig('./results/figs/timecourses/R2_values.pdf')
    plt.close()