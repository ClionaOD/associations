import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from sklearn.linear_model import Ridge
from scipy import stats

def ridge_regress(X,y):
    clf = Ridge(alpha=1, normalize=True)
    clf.fit(X,y)
    coefs = clf.coef_
    return coefs

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
    top_X = pd.DataFrame(counts.nlargest(X,  keep='all'))

    freq_items = top_X.index.tolist()
    
    return freq_items 

def get_coefs(data, sweeps):
    arr = data.values
    coefs = np.zeros((nitems,nitems,len(sweeps)))
    for lag in range(len(sweeps)):
        y = arr[sweeps[-1]:,:]
        X = arr[sweeps[-1] - sweeps[lag] : -sweeps[lag], :]
        coef = ridge_regress(X,y)
        coefs[:,:,lag] = coef

    return coefs

def get_timecourse_coefs(coef_arr, sweeps, nitems=150):
    """
    coef_arr: the 3D array (nitems, nitems, nsweeps) of the mean coefficients
    sweeps: list of sweeps
    nitems: number of frequent items i.e. len(order)
    """
    diags = np.zeros((nitems, len(sweeps)))
    off_diags = np.zeros(((nitems*nitems)-nitems, len(sweeps)))
    
    for lag in range(len(sweeps)):
        lagCoefs = coef_arr[:,:,lag]
        
        d = lagCoefs.diagonal()
        diags[:,lag] = d

        offd = lagCoefs[~np.eye(lagCoefs.shape[0],dtype=bool)].reshape(lagCoefs.shape[0],-1)
        offd = offd.reshape(-1)
        off_diags[:,lag] = offd
    
    return diags, off_diags

def get_timecourse_sigs(coefs, divBy):
    diags = np.zeros((nitems, len(sweeps), divBy))
    off_diags = np.zeros(((nitems*nitems)-nitems, len(sweeps), divBy))

if __name__ == "__main__":
       
    dataPath = './itemsets.pickle'
    orderPath = './freq_order.pickle' #Put to None if the frequent items have not yet been computed
    savePath = './results/coefficients'
    loadPath = '{}/all-betas.pickle'.format(savePath) #Set to None if need to calculate all coefficients

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
        order = most_freq(dataset, nitems)

    #Divide the dataset to allow for mean calculation
    divDataset = divide_dataset(dataset, divBy)

    #Calculate the range of lags to sweep over
    sweeps = np.linspace(1,36000,num=40, dtype=int)
    sweepMins = []
    for sweep in sweeps:
        sweepMins.append(round((sweep*200) / (60*1000)))
    
    #Get coefs over the sweep, separate diagonal and non-diagonal
    if loadPath:
        with open('{}/all-betas.pickle'.format(savePath), 'rb') as f:
            all_betas = pickle.load(f)
    else:
        all_betas = np.zeros((nitems,nitems,len(sweeps),len(divDataset)))
        
        for i in range(len(divDataset)):
            data = encode_dataset(divDataset[i], order)
            coefs = get_coefs(data,sweeps)
            all_betas[:,:,:,i] = coefs

    #Average the coef arrays & get stats
    meanCoefs = np.mean(all_betas, axis=3)

    tstatsCoefs = stats.ttest_1samp(all_betas, 0, axis=3)
    pvalsCoefs = tstatsCoefs.pvalue

    #Separate diagonal
    meanDiags, meanOffs = get_timecourse_coefs(meanCoefs,sweeps,nitems=nitems)
    
    with open('{}/mean-diagonal-coefs.pickle'.format(savePath),'wb') as f:
        pickle.dump(meanDiags,f)

    with open('{}/mean-off-diagonal-coefs.pickle'.format(savePath),'wb') as f:
        pickle.dump(meanOffs,f)

    #plot the timecourses
    thresh = 0.01
    threshold = pvalsCoefs < thresh

    plotDiags = meanDiags.copy()
    plotDiags[~threshold] = 0
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(plotDiags.T)
    ax.set_title('Timecourse of the diagonal values (mean)')
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('coefficients of the diagonal (thresholded at p < {})'.format(thresh))
    plt.savefig('./results/figs/timecourses/diagonal-timecourse_(threshold p < {}).pdf'.format(thresh))
    plt.close()

    plotOffs = meanOffs.copy()
    plotOffs[~threshold] = 0
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(plotOffs.T)
    ax.set_title('Timecourse of the off-diagonal values (mean)')
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('coefficients of the off-diagonal (thresholded at p < {})'.format(thresh))
    plt.savefig('./results/figs/timecourses/off-diagonal-timecourse_(threshold p < {}).pdf'.format(thresh))
    plt.close()

    








