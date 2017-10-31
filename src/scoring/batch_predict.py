"""
Reads a file with one name per line. 
A variable length suffix is removed from each name, and the rest is fed to the name generator 
network as a seed.
"""
import tensorflow as tf
from src.features.char_codec import CharCodec
from src.model.name_generator import NameGenerator
import random
import pandas as pd

def predict(test_seeds, model_name, model_path):
    """
    test_file: file with one name per line
    """
    tf.reset_default_graph()
    names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
    y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)
    model = NameGenerator(model_name=model_name, names=names, next_char_in_name=y)
    saver = tf.train.Saver()
    predictions = []
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for seed in test_seeds:
            res = seed
            initial_seed_offset = len(seed)
            for i in range(CharCodec.max_name_length - len(seed)):
                feats = CharCodec.encode_and_standardize(res).reshape(1, CharCodec.max_name_length)
                prediction = sess.run(model.prediction, feed_dict={names: feats})
                res += " ".join(CharCodec.decode(prediction[0])[i + initial_seed_offset - 1])
                if res[-1] == CharCodec.NAME_END:
                    break
                #seed = CharCodec.decode(prediction[0])[0]
            predictions.append(res)
    return predictions

def get_seeds(names_file):
    """
    :param names: a file with one name per line.
    :return: randomly creates a seed by taking a prefix that's 30% to 50% of the name length.
    """
    names, seeds = [], []
    with open(names_file, "r") as f:
        for line in f:
            line = line.strip()
            for prefix_frac in [0.25, 0.5, 0.75]:
                names.append(line)
                seeds.append(line[:int(len(line) * prefix_frac)])
    return names, seeds

if __name__ == "__main__":
    races = ["hispanic", "indian", "african_american", "caucasian", "all_races"]
    #races = ["caucasian"]
    names, seeds = get_seeds("../../data/test.txt")
    print(names)
    print(seeds)
    res = {}
    res["names"] = names
    res["seeds"] = seeds
    #seeds = ["undertaker"[:i] for i in range(1, 5)]
    #seeds = ["hash"]
    for race in races:
        res[race] = predict(test_seeds=seeds, model_name=race, model_path="../../models/{0}_2000_eps".format(race))
    pd.DataFrame.from_dict(res).to_csv("../../data/predictions.csv", index=False)
    print(res)