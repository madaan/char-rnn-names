import tensorflow as tf
from datetime import datetime
from src.features.char_codec import CharCodec
import logging
import sys
"""

"""

    
logging.getLogger().setLevel(logging.INFO)
model_path = "../../models/single_lstm_1000_epochs_3.ckpt"

n_epochs = 1000
is_training = True
names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)
saver = tf.train.Saver()
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
        
            feats = CharCodec.encode_and_standardize(res).reshape(1, max_name_length)
            prediction, ops = sess.run([best_chars, outputs], feed_dict={names: feats})
            res += " ".join(decode_classes_to_ name(prediction[0])[i + initial_seed_offset - 1])
            if res[-1] == NAME_END:
                break
            seed = decode_classes_to_name(prediction[0])[0]
        print(res)