import logging
from datetime import datetime
import tensorflow as tf
from src.features.char_codec import CharCodec
from src.features.names_dataset import NameGenerationCharLevelDataset
from src.model.name_generator import NameGenerator
from tensorflow.contrib.tensorboard.plugins import projector
import os

n_epochs = 50

names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)

model = NameGenerator(names=names, next_char_in_name=y)


model_path = "../../models/single_lstm_1000_epochs_6.ckpt"

model_name = "single_lstm.ckpt"

##tensorboard

### log cross entropy
logdir = "../../logs/run-{0}".format(datetime.now().strftime("%Y%m%d%H%M%S"))
xentropy_summary = tf.summary.scalar("Cross_Entropy_Loss", model.loss)

### plot embeddings
embedding_var = model.char_embedding_matrix
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.join('char_metadata.txt')
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
#projector.visualize_embeddings(file_writer, config)


with tf.Session() as sess:

    dataset = NameGenerationCharLevelDataset(["../../data/names.txt"])
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    for i in range(n_epochs):
        start_time = datetime.now()
        for j, (features, labels) in enumerate(dataset.iterator(num_batches=20)):
             sess.run([model.train_op], feed_dict= { names : features, y: labels})

        ls, summary = sess.run([model.loss, xentropy_summary], feed_dict={names: features,   y: labels})
        print("epoch {0}, loss = {1}, time = {2}".format(i, ls, datetime.now() - start_time))

        #The
        file_writer.add_summary(summary, i)
        saver.save(sess, os.path.join(logdir, model_name), global_step=i)

    saver.save(sess, model_path)

file_writer.close()