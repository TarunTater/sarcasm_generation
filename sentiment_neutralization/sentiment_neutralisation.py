"""Sentiment Neutralisation Code."""

import os
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import OneHotEncoder
import neutralisation_params

def main():
    """Function for removing sentiment words from a given text"""

    data = pd.read_csv('../data/dataSampleTest.csv')
    data.columns = ["sarcasmText", "text"] # Keeping only the neccessary columns

    stop = stopwords.words('english')
    data['text_without_stopwords'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
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
    enc = OneHotEncoder(handle_unknown='ignore', n_values=max_features, sparse=False)
    x_train_one_hot = enc.fit_transform(X)
    x_train_one_hot = np.reshape(x_train_one_hot, (X.shape[0], maxlen, max_features))

    model_name = neutralisation_params.model_name
    trained_model = os.path.join(model_dir, model_name) + "_final.hdf5"

    # SAVE_FEATURES_PREFIX = neutralisation_params.SAVE_FEATURES_PREFIX
    model = load_model(trained_model, custom_objects={'SeqSelfAttention': SeqSelfAttention})
    # batch_size = neutralisation_params.batch_size

    feat_dir = neutralisation_params.feature_dir
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    print(model.summary())

    dense_model = Model(inputs=model.input, outputs=model.get_layer('seq_self_attention_1').output)
    dense_feature, attn_weight = dense_model.predict(X)
    # attn_weight_collapsed = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))

    new_data_all = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))
    for i in range(0, attn_weight.shape[0]):
        current_max_array = attn_weight[i].max(0)
        temp_list = []
        for k in range(0, current_max_array.shape[0]):
            if current_max_array[k] != 0:
                temp_list.append(current_max_array[k])

        current_mean = np.mean(temp_list)
        current_std = np.std(temp_list)
        num_higher = current_mean + 1*(current_std)
        num_lower = current_mean - 1.5*(current_std)
        high_outlier = (current_max_array <= num_higher).astype(int)
        low_outlier = (current_max_array > num_lower).astype(int)
        context_ones = high_outlier*low_outlier
        new_data = X[i] * context_ones
        new_data_all[i] = new_data

    def sequence_to_text(list_of_indices):
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words

    my_texts = list(map(sequence_to_text, new_data_all))
    correct_sent = list(data["text"])
    all_sentences = []
    new_training_output = []
    for i in range(0, len(my_texts)):
        each_new = [x for x in my_texts[i] if x is not None]
        each_new = " ".join(each_new)
        if each_new != "":
            rem = correct_sent[i]
            new_training_output.append(rem)
            all_sentences.append(each_new)
        else:
            print(i, correct_sent[i])

    print(len(all_sentences), len(new_training_output))
    for i in range(0, len(new_training_output)):
        print(all_sentences[i], " ----------- ", new_training_output[i])

if __name__ == '__main__':
	main()
