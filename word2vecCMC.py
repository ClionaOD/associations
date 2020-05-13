import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

chosenCategs = ['gown', 'hair', 'suit', 'coat', 'tie', 'shirt', 'sunglasses', 'shoe', 'screen', 'computer', 'table', 'food', 'restaurant', 'glass', 'alcohol', 'wine', 'lamp', 'couch', 'chair', 'closet', 'piano', 'pillow', 'desk', 'window', 'bannister']
modelPath = '/home/CUSACKLAB/clionaodoherty/GoogleNews-vectors-negative300.bin'

#load wordnet model
model = KeyedVectors.load_word2vec_format(modelPath, binary=True, limit=500000)

sim_df = pd.DataFrame(index=chosenCategs, columns=chosenCategs, dtype=float)

for w1, j in sim_df.iteritems():
    for w2 in j.index:
        sim_df.loc[w1][w2] = float(model.distance(w1,w2))
        sim_df.loc[w2][w1] = float(model.distance(w1,w2))

