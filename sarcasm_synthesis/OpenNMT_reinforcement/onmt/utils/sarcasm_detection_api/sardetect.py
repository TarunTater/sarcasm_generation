from __future__ import print_function, division
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import pickle,sys

### Directory of current exec###
dir_path = os.path.dirname(os.path.realpath(__file__))

#####################Get the model builder code ready####################

# Define some parameters
EMBEDDING_DIM = 100
HIDDEN_UNITS = 150
ATTENTION_UNITS = 50
KEEP_PROB = 0.8
DELTA = 0.5
SHUFFLE = False
max_tweet_length = 47

vocab = pickle.load(open(dir_path+"/model/vocab.pkl","rb"))

vocabulary_size = len(vocab.keys())+1 #add one for UNK

# Set the sequence length
SEQUENCE_LENGTH = max_tweet_length


# This is piece of code is Copyright (c) 2017 to Ilya Ivanov and grants permission under MIT Licence
# https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
# Implementation as proposed by Yang et al. in "Hierarchical Attention Networks for Document Classification" (2016)
def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def build_attention_model():
    # Different placeholders
    with tf.name_scope('Inputs'):
        batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
        target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
        seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
        keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

    # Embedding layer
    with tf.name_scope('Embedding_layer'):
        embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
        tf.summary.histogram('embeddings_var', embeddings_var)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

    # (Bi-)RNN layer(-s)
    rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_UNITS), GRUCell(HIDDEN_UNITS),
                            inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    tf.summary.histogram('RNN_outputs', rnn_outputs)

    # Attention layer
    with tf.name_scope('Attention_layer'):
        attention_output, alphas = attention(rnn_outputs, ATTENTION_UNITS, return_alphas=True)
        tf.summary.histogram('alphas', alphas)

    # Dropout
    drop = tf.nn.dropout(attention_output, keep_prob_ph)

    # Fully connected layer
    with tf.name_scope('Fully_connected_layer'):
        W = tf.Variable(
            tf.truncated_normal([HIDDEN_UNITS * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
        b = tf.Variable(tf.constant(0., shape=[1]))
        y_hat = tf.nn.xw_plus_b(drop, W, b)
        y_hat = tf.squeeze(y_hat)
        pred = tf.sigmoid(y_hat)
        #pred = tf.round(tf.sigmoid(y_hat))
        tf.summary.histogram('W', W)

    with tf.name_scope('Metrics'):
        # Cross-entropy loss and optimizer initialization
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        # Accuracy metric
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    # Batch generators
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    saver = tf.train.Saver()
    return batch_ph, target_ph, seq_len_ph, keep_prob_ph, alphas,pred, session_conf, saver

##############Now prepare test scenario###########

MODEL_PATH = dir_path+"/model/"
batch_ph, target_ph, seq_len_ph, keep_prob_ph, alphas,pred, session_conf, saver = build_attention_model()


sess = tf.Session()
saver.restore(sess, MODEL_PATH)
print ("Sarcasm Detection Model Restored")

def get_sentence_score(sent,max_len=47):
    sent = " ".join(sent.split()[:45])
    array1 = [0]*max_len
    i = 0
    for w in sent.split():
        array1[i] = vocab.get(w.lower(),0)
        i+=1
    array2 = np.array(array1)
    #print (array2.shape)
    seql = [len(array1)]#len(sent.split())
    x_batch = [array2]
    #print (x_batch)
    y_batch = [0]
    out = sess.run([pred],feed_dict={batch_ph: x_batch,target_ph: y_batch,seq_len_ph: seql,keep_prob_ph: 1.0})
    return out[0]

def get_batch_reward_score(sents):
    scores = []
    for s in sents:
        scores.append(1-get_sentence_score(s))
    return sum(scores)/float(len(scores))
     

if __name__=="__main__":
    while True:
        print ("Enter Sent")
        inp = input()
        print (get_sentence_score(inp))

