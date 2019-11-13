import pandas as pd
import os
import pickle
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import OneHotEncoder
import neutralisation_params

def main():

    data = pd.read_csv('../data/dataSampleTest.csv')
    data.columns = ["sarcasmText", "text"] # Keeping only the neccessary columns

    stop = stopwords.words('english')
    data['text_without_stopwords'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    data['text_without_stopwords'] = data['text_without_stopwords'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: x.lower())

    model_dir = neutralisation_params.RESULT_DIR
    token_path = os.path.join(model_dir, neutralisation_params.token_file_path)
    complete_token_path = token_path + ".pickle"
    with open(complete_token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    maxlen = neutralisation_params.maxlen
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X, maxlen=maxlen)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(X.shape)

    max_features = neutralisation_params.max_features
    enc = OneHotEncoder(handle_unknown='ignore',n_values=max_features,sparse=False)
    X_train_one_hot = enc.fit_transform(X)
    X_train_one_hot = np.reshape(X_train_one_hot,(X.shape[0],maxlen,max_features))

    '''
    Extract features from the data
    '''
    model_name = neutralisation_params.model_name
    TRAINED_MODEL = os.path.join(model_dir, model_name) + "_final.hdf5"

    # SAVE_FEATURES_PREFIX = neutralisation_params.SAVE_FEATURES_PREFIX
    model = load_model(TRAINED_MODEL,custom_objects={'SeqSelfAttention': SeqSelfAttention})
    batch_size = neutralisation_params.batch_size

    feat_dir = neutralisation_params.feature_dir
    if(not os.path.exists(feat_dir)):
        os.makedirs(feat_dir)
    print(model.summary())

    dense_model = Model(inputs=model.input, outputs=model.get_layer('seq_self_attention_1').output)
    dense_feature, attn_weight = dense_model.predict(X)
    attn_weight_collapsed = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))

    newDataAll = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))
    for i in range(0, attn_weight.shape[0]):
        currentMaxArray = attn_weight[i].max(0)
        tempList = []
        for k in range(0, currentMaxArray.shape[0]):
            if(currentMaxArray[k] != 0):
                tempList.append(currentMaxArray[k])
        lenToDivide = len(tempList)
        tempArr = np.array(tempList)
        current_mean = np.mean(tempList)
        current_std = np.std(tempList)
        numHigher = current_mean + 1*(current_std)
        numLower = current_mean - 1.5*(current_std)
        highOutlier = (currentMaxArray <= numHigher ).astype(int)
        lowOutlier = (currentMaxArray > numLower).astype(int)
        contextOnes = highOutlier*lowOutlier
        newData = X[i] * contextOnes
        newDataAll[i] = newData

    def sequence_to_text(list_of_indices):
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return(words)

    my_texts = list(map(sequence_to_text, newDataAll))
    correctSent = list(data["text"])
    allSentences = []
    newTrainingOutput = []
    for i in range(0, len(my_texts)):
        eachNew = [x for x in my_texts[i] if x is not None]
        eachNew = " ".join(eachNew)
        if(eachNew != ""):
            rem = correctSent[i]
            newTrainingOutput.append(rem)
            allSentences.append(eachNew)
        else:
            print(i, correctSent[i])
    len(allSentences), len(newTrainingOutput)

    for i in range(0, len(newTrainingOutput)):
        print(allSentences[i], " ----------- ", newTrainingOutput[i])

if __name__ == '__main__':
	main()
