import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from sklearn.linear_model import Ridge

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

def get_timecourse_coefs(data, sweeps, mode=['diagonal','off-diagonal']):
    arr = data.values
    
    if mode == 'diagonal':
        diags = np.zeros((nitems, len(sweeps))
        for lag in range(len(sweeps)):
            y = arr[sweeps[-1]:,:]
            X = arr[sweeps[-1] - sweeps[lag] : -sweeps[lag], :]
            coef = ridge_regress(X,y)
            d = coef.diagonal()
            diags[:,lag] = d

        return diags
    
    elif mode == 'off-diagonal':
        off-diags = np.zeros(((nitems*nitems)-nitems, len(sweeps)))
        
        for lag in range(len(sweeps)):
            y = arr[sweeps[-1]:,:]
            X = arr[sweeps[-1] - sweeps[lag] : -sweeps[lag], :]
            coef = ridge_regress(X,y)

            #remove diagonals and flatten
            offd = coef[~np.eye(coef.shape[0],dtype=bool)].reshape(coef.shape[0],-1)
            offd = offd.reshape(-1)
            off-diags[:,lag] = offd
        
        return off-diags


if __name__ == "__main__":
       
    dataPath = './itemsets.pickle'
    orderPath = './freq_order.pickle' #Put to None if the frequent items have not yet been computed
    savePath = './results/coefficients'

    nitems = 150
    
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
    divDataset = divide_dataset(dataset, 16)

    #Calculate the range of lags to sweep over
    sweeps = np.linspace(1,36000,num=40, dtype=int)
    sweepMins = []
    for sweep in sweeps:
        sweepMins.append(round((sweep*200) / (60*1000)))
    
    #Get coefs of the diagonals over the sweep
    diagonal_betas = np.zeros((nitems,len(sweeps),len(divDataset)))
    off_betas = np.zeros(((nitems*nitems)-nitems, len(sweeps),len(divDataset)))
    for i in range(len(divDataset)):
        data = encode_dataset(divDataset[i], order)
        diagonal_betas[:,:,i] = get_timecourse_coefs(data,sweeps,mode='diagonal')
        off_betas[:,:,i] = get_timecourse_coefs(data,sweeps,mode='off-diagonal')

    #Average and save out diagonal values
    meanDiags = np.mean(diagonal_betas, axis=2)
    
    with open('{}/mean-diagonal-coefs.pickle'.format(savePath),'rb') as f:
        pickle.dump(meanDiags,f)

    #Average and save out off-diagonal values
    meanOffs = np.mean(off_betas, axis=2)

    with open('{}/mean-off-diagonal-coefs.pickle'.format(savePath),'rb') as f:
        pickle.dump(meanOffs,f)

    #plot the timecourses
    threshold = 0.001
    
    plotDiags = np.delete(meanDiags, np.where(meanVals[:,0] < threshold)[0], 0)
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(plotDiags.T)
    ax.set_title('Timecourse of the diagonal values (mean)')
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('coefficients of the diagonal (thresholded at Β > {})'.format(threshold))
    plt.savefig('./results/figs/timecourses/diagonal-timecourse_(threshold {}).pdf'.format(threshold))
    plt.close()

    plotOffs = np.delete(meanOffs, np.where(meanVals[:,0] < threshold)[0], 0)
    fig, ax = plt.subplots(figsize=[25,13])
    ax.plot(plotOffs.T)
    ax.set_title('Timecourse of the off-diagonal values (mean)')
    ax.set_xlabel('minutes')
    ax.set_xticks(range(40))
    ax.set_xticklabels(sweepMins)
    ax.set_ylabel('coefficients of the off-diagonal (thresholded at Β > {})'.format(threshold))
    plt.savefig('./results/figs/timecourses/off-diagonal-timecourse_(threshold {}).pdf'.format(threshold))
    plt.close()

    








