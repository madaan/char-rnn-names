from datetime import datetime
import tensorflow as tf
from src.features.char_codec import CharCodec
from src.features.names_dataset import NameGenerationCharLevelDataset
from src.model.name_generator import NameGenerator
import os
import sys
import json

LOG_DIR = "../../logs"
SESSION_DIR = "../../models"

def train(names_filepaths, log_base_dir, model_dir, model_name, n_epochs=1000):

    names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
    y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)
    model = NameGenerator(model_name=model_name, names=names, next_char_in_name=y)

    ##tensorboard

    ### log cross entropy
    logdir = "{0}/run-{1}-{2}".format(log_base_dir, model_name, datetime.now().strftime("%Y%m%d%H%M%S"))
    xentropy_summary = tf.summary.scalar("Cross_Entropy_Loss", model.loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        dataset = NameGenerationCharLevelDataset(names_filepaths)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(n_epochs):
            start_time = datetime.now()
            for j, (features, labels) in enumerate(dataset.iterator(num_batches=20)):
                 sess.run([model.train_op], feed_dict= { names : features, y: labels})

            #get loss for one chunk.
            ls, summary = sess.run([model.loss, xentropy_summary], feed_dict={names: features, y: labels})
            print("epoch {0}, loss = {1}, time = {2}".format(i, ls, datetime.now() - start_time))
            file_writer.add_summary(summary, i)
            saver.save(sess, os.path.join(logdir, model_name), global_step=i)

        saver.save(sess, "{0}/{1}_{2}_eps".format(model_dir, model_name, n_epochs))

    file_writer.close()

if __name__ == "__main__":
    """
    datasets = {
        "datasets" : [{
            "name" : "hispanic",
            "data" : "../../data/hispanic_names.txt"
        },{
            "name" : "caucasian",
            "data" : "../../data/caucasian_names.txt"
        },{
            "name" : "indian",
            "data" : "../../data/indian_names.txt"
        },{
            "name" : "african_american",
            "data" : "../../data/african_american_names.txt"
        }, {
            "name": "all_races",
            "data": "../../data/all_races_names.txt"
        }]
    }
"""
    with open(sys.argv[1], "r") as f:
        train_conf = json.load(f)
    for dataset in train_conf["datasets"]:
        print("Processing {0}".format(dataset["name"]))
        train(names_filepaths=[dataset["data"]],
              log_base_dir=train_conf["log_dir"],
              model_dir=train_conf["model_dir"],
              model_name=dataset["name"], n_epochs=train_conf["n_epochs"])