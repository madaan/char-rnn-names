'''

'''
import tensorflow as tf
from datetime import datetime
from src.features.char_codec import CharCodec
import logging

#from code.gender.read_data import *

"""

"""

class NameGenerator:

    """
    Setup the architecture
    """
    n_embeddings = 20
    n_layers = 3
    n_hidden = 150
    num_epochs = 1000

    def __init__(self, names, next_char_in_name):

        embeddings = tf.Variable(tf.random_uniform([CharCodec.vocab_size, self.n_embeddings], -1.0, 1.0))
        embedded_names = tf.nn.embedding_lookup(embeddings, names) #(?, seq_length, n_embeddings)

        """
        Next, we define one recurrent cell. The one that takes a vector of dimension n_embeddings as the input and
        generates a vector of size n_hidden as the output. This is for a single step. 
        """
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden) #(1, n_hidden)

        """
        Dynamic RNN helps us in repeating this LSTM cell as many times as needed. In our case, we need one LSTM 
        cell per input sequence element. Thus, we feed in our embedded names vector, which has the
        dimensions (?, seq_length, n_embeddings). Note that the lstm_cell is a mapping from n_embeddings -> n_hidden.
        The dymanic_rnn function basically creates max_name_length of these embeddings, and emits the following two 
        vectors:
        - outputs (?, seq_length, n_hidden)
        - states (?, n_hidden)
        There is one output per lstm_cell, and there is final state.
        """
        rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=embedded_names, dtype=tf.float32)

        """
        We feed the per step output to a simple dense layer that predicts the next character. 
        """
        dense_layer_in = tf.reshape(rnn_outputs, shape=[-1, self.n_hidden]) #(?, n_hidden)
        dense_layer_out = tf.layers.dense(dense_layer_in, CharCodec.vocab_size) #(?, vocab_size)

        outputs = tf.reshape(dense_layer_out, shape=[-1, CharCodec.max_name_length, CharCodec.vocab_size]) #(?, seq_length, vocab_size)

        logits = tf.reshape(outputs, shape=[-1, vocab_size]) #(?, vocab_size)

        softmax_ = tf.nn.softmax(logits)

        #setup losses
        loss_unreduced = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        loss = tf.reduce_mean(loss_unreduced)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
        saver = tf.train.Saver()
        #prediction
        best_chars = tf.argmax(outputs, axis=2)

logging.getLogger().setLevel(logging.INFO)
import sys
model_path = "../../models/single_lstm_1000_epochs_3.ckpt"

names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if is_training:
        for i in range(num_epochs):
            start_time = datetime.now()
            for j, (features, labels) in enumerate(name_generation_features(num_batches=20)):
                 sess.run([train_op], feed_dict= { names : features, y: labels})

            ls = sess.run([loss], feed_dict={names: features,   y: labels})
            print("epoch {0}, loss = {1}, time = {2}".format(i, ls, datetime.now() - start_time))

        saver.save(sess, model_path)
    else:
        saver.restore(sess, model_path)

    while True:
        seed = input("> ")
        res = seed
        desired_length = 20
        initial_seed_offset = len(seed)
        for i in range(desired_length):
            feats = get_features_given_name(res).reshape(1, max_name_length)
            prediction, ops = sess.run([best_chars, outputs], feed_dict={names: feats})
            res += " ".join(decode_classes_to_name(prediction[0])[i + initial_seed_offset - 1])
            if res[-1] == NAME_END:
                break
            seed = decode_classes_to_name(prediction[0])[0]
        print(res)