import numpy as np
from src.features.char_codec import CharCodec


class NameGenerationCharLevelDataset:
    """
    Reads the names dataset from the disk and creates an iterator over (names, names_shifted_left).
    """
    def __init__(self, name_filepaths):
        name_filepaths = name_filepaths
        names = []
        for name_filepath in name_filepaths:
            with open(name_filepath, "r") as f:
                for line in f:
                    names.append(line.strip())

        names_shifted_left = [[] for i in range(len(names))]
        # > names_shifted_left = []
        for i, name in enumerate(names):
            names[i] = CharCodec.encode_and_standardize(name)
            # > temp = np.roll(names[i], -1)
            # > temp[-1] = CharEncoding.INVALID_CHAR_CLASS
            for c in names[i][1:]:
                names_shifted_left[i].append(c)
            # > temp.append(CharEncoding.INVALID_CHAR_CLASS)
            # > names_shifted_left.append(temp)
            names_shifted_left[i].append(CharCodec.INVALID_CHAR_CLASS)
        self.m = len(names)
        self.names = np.array(names)
        self.names_shifted_left = np.array(names_shifted_left)

    def iterator(self, num_batches):
        """

        :param num_batches:
            The number of batches
        :return:
        Iterates over the given files, and collects all the names in a list called name.
        For each name in names, generates a tuple (name, name_left_shifted_by_one).
        Where both name and name_left_shifted_by_one are encoded and standardized by the operations defined
        in CharEncoder.
        The intended usage is for "name" to be used as the input sequence, and "name_left_shifted_by_one" to be
        used as the output sequence so that the network can learn to predict the next character.

        The operations commented out with a "#>" are cleaner, but slower by a lot.
        """
        p = np.random.permutation(self.m)
        names = self.names[p, :]
        names_shifted_left = self.names_shifted_left[p, :]
        batch_size = self.m // num_batches

        for i in range(num_batches):
            yield (names[i * batch_size:(i + 1) * batch_size, :],
                   names_shifted_left[i * batch_size:(i + 1) * batch_size, :].reshape(-1))


"""

filenames = ["../../data/names.txt"]

dataset = tf.contrib.data.TextLineDataset(filenames)


#dataset = dataset.map(lambda x: tf.decode_raw(x))


#dataset = dataset.map(lambda x: tf.py_func(CharEncoding.encode_and_standardize, [x], [tf.int32 for _ in range(CharEncoding.max_name_length)]))

dataset = dataset.map(lambda x: tf.py_func(CharEncoding.encode_and_standardize, [x], [tf.int32]))
#iterator = dataset.make_initializable_iterator()

batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()

print(sess.run(next_element)[0].shape)
sess.close()
next_element = iterator.get_next()
# Compute for 100 epochs.
with tf.Session() as sess:
    for _ in range(100):
      sess.run(iterator.initializer)
      while True:
        try:
          ne = sess.run(next_element)
          print(ne)
          #print(CharEncoding.encode_and_standardize(str(ne)))
        except tf.errors.OutOfRangeError:
          break
"""