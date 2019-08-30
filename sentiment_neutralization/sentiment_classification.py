#!/usr/bin/env python
# coding: utf-8

# In[1]:



from __future__ import print_function

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


# In[2]:


import re
import time
import pandas as pd
import numpy as np
import json
from pprint import pprint
import os


# In[3]:



from numpy import array
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import load_model, Model



# In[ ]:





# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import pickle

# In[5]:
data_dir = os.environ.get("DATA_DIR", "./")

model_dir = os.environ.get("RESULT_DIR", "./")

# data = pd.read_csv('dlassData_feb16.csv')
data = pd.read_csv(os.path.join(data_dir,'sentiment_data.csv'), encoding = "ISO-8859-1")

data.columns = ["text", "label"]


# In[6]:


data['text'] = data['text'].apply(lambda x: x.lower())


# In[7]:


def classificationLabels(row):
    if(row["label"] == 1):
        return "Negative"
    return "Positive"
data["sentiment"] = data.apply(classificationLabels, axis=1)


# In[8]:


data.drop("label", axis=1,inplace=True)


# In[9]:


data


# In[10]:


data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


# In[11]:


print(data[data['sentiment'] == 'Positive'].size)
print(data[data['sentiment'] == 'Negative'].size)

max_fatures = 19676
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, maxlen=30)


# In[12]:


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
# print(reverse_word_map[1])

# In[14]:




# In[15]:
token_path = os.path.join(model_dir, "token_mar1_v3.pickle")

with open(token_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[16]:


print(X.shape)


# In[17]:


data['sentiment'].value_counts()


# In[21]:





# In[19]:


embed_dim = 128


# In[22]:


inputs = Input(shape=(30,))

emb1 = Embedding(max_fatures, embed_dim, mask_zero=True)(inputs)

lstm1 = LSTM(200, return_sequences=True)(emb1)
lstmOut, attWeights = SeqSelfAttention(attention_activation='sigmoid', return_attention=True)(lstm1)
lstm2 = LSTM(150, return_sequences=False, trainable = False)(lstmOut)
# d1 = Dense(50, activation='sigmoid')(lstm2)
# d2 = Dense(10, activation='sigmoid')(d1)
outputs = Dense(2, activation='sigmoid')(lstm2)
model = Model(inputs=[inputs], outputs=outputs)


# In[23]:


model.summary()


# In[26]:


Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 32
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_name = "v3" + "_" + str(embed_dim)
filepath = os.path.join(model_dir, model_name) + "_intermediate.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train,batch_size=batch_size, epochs=10, validation_data=(X_test, Y_test),verbose=1, callbacks=callbacks_list)

print("Training has finished. Model save at ", os.path.join(model_dir, model_name) + "_final.hdf5")
model.save(os.path.join(model_dir, model_name) + "_final.hdf5")
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

print('Test score:', score)
print('Test accuracy:', acc)

# model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 1)
