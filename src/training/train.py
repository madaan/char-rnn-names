import logging
from datetime import datetime
import tensorflow as tf
from src.features.char_codec import CharCodec
from src.features.names_dataset import NameGenerationCharLevelDataset
from src.model.name_generator import NameGenerator
from tensorflow.contrib.tensorboard.plugins import projector
import os
tf.logging.set_verbosity(tf.logging.INFO)

LOG_DIR = "../../logs"
SESSION_DIR = "../../models"

def train(names_filepaths, model_name, n_epochs=1000):

    names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
    y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)
    model = NameGenerator(names=names, next_char_in_name=y)

    ##tensorboard

    ### log cross entropy
    logdir = "{0}/run-{1}-{2}".format(LOG_DIR, model_name, datetime.now().strftime("%Y%m%d%H%M%S"))
    xentropy_summary = tf.summary.scalar("Cross_Entropy_Loss", model.loss)

    ### plot embeddings
    """
    embedding_var = model.char_embedding_matrix
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join('char_metadata.txt')
    """
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    #projector.visualize_embeddings(file_writer, config)


    with tf.Session() as sess:

        dataset = NameGenerationCharLevelDataset(names_filepaths)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        for i in range(n_epochs):
            start_time = datetime.now()
            for j, (features, labels) in enumerate(dataset.iterator(num_batches=20)):
                 sess.run([model.train_op], feed_dict= { names : features, y: labels})

            ls, summary = sess.run([model.loss, xentropy_summary], feed_dict={names: features,   y: labels})
            print("epoch {0}, loss = {1}, time = {2}".format(i, ls, datetime.now() - start_time))
            file_writer.add_summary(summary, i)
            saver.save(sess, os.path.join(logdir, model_name), global_step=i)

        saver.save(sess, "{0}/{1}_{2}_eps".format(SESSION_DIR, model_name, n_epochs))

    file_writer.close()

if __name__ == "__main__":
    x = {
        "datasets" : [{
            "name" : "hispanic",
            "data" : "../../data/hipanic_names.txt"
        }]
    }

    train(names_filepaths=["../../data/hispanic_names.txt"], model_name="hispanic", n_epochs=100)