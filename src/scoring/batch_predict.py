"""
Reads a file with one name per line. 
A variable length suffix is removed from each name, and the rest is fed to the name generator 
network as a seed.
"""
import tensorflow as tf
from features.char_codec import CharCodec
def predict(test_file):
    """
    test_file: file with one name per line
    """
    names = tf.placeholder(tf.int32, shape=(None, CharCodec.max_name_length), name="input")
    y = tf.placeholder(dtype=tf.int32, shape=[None])  # (?)
    model = NameGenerator(names=names, next_char_in_name=y)
    model_path = "../../models/single_lstm_1000_epochs_4.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        with open(test_file, "r") as f:
            for line in f:
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

