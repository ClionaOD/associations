import pickle

with open('/home/CUSACKLAB/clionaodoherty/associations/diagonal_over_0.01.pickle','rb') as f:
    items=pickle.load(f)

with open('/home/CUSACKLAB/clionaodoherty/associations/freq_order.pickle','rb') as f:
    order=pickle.load(f)dsa


for x in items:

    print(order[x]) #, order[y])