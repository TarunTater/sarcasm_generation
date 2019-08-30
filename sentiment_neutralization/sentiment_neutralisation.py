

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Bidirectional, SpatialDropout1D
from keras.layers import LSTM
from keras.datasets import imdb
from keras_self_attention import SeqSelfAttention
from numpy import array
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

import re
import time
import pandas as pd
import numpy as np
import json
from pprint import pprint
import os


from numpy import array
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import load_model, Model


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
import pickle
from sklearn.preprocessing import OneHotEncoder

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

data = pd.read_csv('../dataSampleTest.csv')
# Keeping only the neccessary columns
data.columns = ["sarcasmText", "text"]

stop = stopwords.words('english')
data['text_without_stopwords'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data['text_without_stopwords'] = data['text_without_stopwords'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: x.lower())


with open('token.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, maxlen=30)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
print(X.shape)

max_fatures =20000


enc = OneHotEncoder(handle_unknown='ignore',n_values=max_fatures,sparse=False)
X_train_one_hot = enc.fit_transform(X)
X_train_one_hot = np.reshape(X_train_one_hot,(X.shape[0],30,max_fatures))

'''
Extract features from the data
'''

from keras.models import load_model, Model

TRAINED_MODEL = "model_final.hdf5"

SAVE_FEATURES_PREFIX = "m1_features"
model = load_model(TRAINED_MODEL,custom_objects={'SeqSelfAttention': SeqSelfAttention})
batch_size = 32

feat_dir = "./features"
if(not os.path.exists(feat_dir)):
    os.makedirs(feat_dir)


print(model.summary())

dense_model = Model(inputs=model.input, outputs=model.get_layer('seq_self_attention_1').output)
dense_feature, attn_weight = dense_model.predict(X)

attn_weight_collapsed = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))


newDataAll = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))
for i in range(0, attn_weight.shape[0]):
    currentMaxArray = attn_weight[i].max(0)
#     print(currentMaxArray)
    tempList = []
    for k in range(0, currentMaxArray.shape[0]):
        if(currentMaxArray[k] != 0):
            tempList.append(currentMaxArray[k])
    lenToDivide = len(tempList)
#     lenToDivide = 19
    tempArr = np.array(tempList)
    current_mean = np.mean(tempList)
    current_std = np.std(tempList)
#     print(current_mean, current_std)

    numHigher = current_mean + 1*(current_std)
    numLower = current_mean - 1.5*(current_std)
    highOutlier = (currentMaxArray <= numHigher ).astype(int)
    lowOutlier = (currentMaxArray > numLower).astype(int)
#     print(highOutlier, lowOutlier)

    contextOnes = highOutlier*lowOutlier

    newData = X[i] * contextOnes
#     print(newData)

    newDataAll[i] = newData
#     print(attn_weight_collapsed[i])


my_texts = list(map(sequence_to_text, newDataAll))
correctSent = list(data["text"])
allSentences = []
newTrainingOutput = []
for i in range(0, len(my_texts)):
    eachNew = [x for x in my_texts[i] if x is not None]
    eachNew = " ".join(eachNew)
    if(eachNew != ""):
        rem = correctSent[i]
#         rem = rem[2:-2]
        newTrainingOutput.append(rem)
        allSentences.append(eachNew)
    else:
        print(i, correctSent[i])
len(allSentences), len(newTrainingOutput)

for i in range(0, len(newTrainingOutput)):
    print(allSentences[i], " ----------- ", newTrainingOutput[i])
