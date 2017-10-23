'''

'''
import tensorflow as tf
from datetime import datetime
from src.features.char_codec import CharCodec
#from code.gender.read_data import *
seq_length = CharCodec.max_name_length
n_embeddings = 20
n_layers = 3
n_hidden = 150
num_epochs = 1000
is_training = True

vocab_size = 26 + 3

names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")

embeddings = tf.Variable(tf.random_uniform([vocab_size, n_embeddings], -1.0, 1.0))

embedded_names = tf.nn.embedding_lookup(embeddings, names) #(?, seq_length, n_embeddings)

"""
recurrent_cell = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden) for layer in range(n_layers)]
multi_recurrent_cell = tf.nn.rnn_cell.MultiRNNCell(recurrent_cell)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_recurrent_cell, inputs=embedded_names, dtype=tf.float32)
"""
recurrent_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden)
rnn_outputs, states = tf.nn.dynamic_rnn(recurrent_cell, inputs=embedded_names, dtype=tf.float32)

#outputs (?, seq_length, n_hidden)
#states (?, 150)

dense_layer_in = tf.reshape(rnn_outputs, shape=[-1, n_hidden]) #(?, n_hidden)
dense_layer_out = tf.layers.dense(dense_layer_in, vocab_size) #(?, vocab_size)
outputs = tf.reshape(dense_layer_out, shape=[-1, seq_length, vocab_size]) #(?, seq_length, vocab_size)

y = tf.placeholder(dtype=tf.int32, shape=[None]) #(?)
logits = tf.reshape(outputs, shape=[-1, vocab_size]) #(?, vocab_size)

smax = tf.nn.softmax(logits)

#setup losses
loss_unreduced = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
loss = tf.reduce_mean(loss_unreduced)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
saver = tf.train.Saver()
#prediction
best_chars = tf.argmax(outputs, axis=2)

import logging
logging.getLogger().setLevel(logging.INFO)
import sys
model_path = "../../models/single_lstm_1000_epochs_3.ckpt"
config = tf.ConfigProto(
        #device_count = {'GPU': 0}
    )

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