import tensorflow as tf
from datetime import datetime
from src.features.char_codec import CharCodec
import logging
import sys
from src.training.name_generator import NameGenerator
from src.features.names_dataset import NameGenerationCharLevelDataset
"""

"""

    
logging.getLogger().setLevel(logging.INFO)
model_path = "../../models/single_lstm_1000_epochs_4.ckpt"

n_epochs = 3000
is_training = False
names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)
model = NameGenerator(names=names, next_char_in_name=y)
dataset = NameGenerationCharLevelDataset(["../../data/names.txt"])

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if is_training:
        for i in range(n_epochs):
            start_time = datetime.now()
            for j, (features, labels) in enumerate(dataset.iterator(num_batches=20)):
                 sess.run([model.train_op], feed_dict= { names : features, y: labels})

            ls = sess.run([model.loss], feed_dict={names: features,   y: labels})
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
            feats = CharCodec.encode_and_standardize(res).reshape(1, CharCodec.max_name_length)
            prediction = sess.run(model.prediction, feed_dict={names: feats})
            res += " ".join(CharCodec.decode(prediction[0])[i + initial_seed_offset - 1])
            if res[-1] == CharCodec.NAME_END:
                break
            seed = CharCodec.decode(prediction[0])[0]
        print(res)
