import tensorflow as tf

from src.features.char_codec import CharCodec


class NameGenerator:

    """
    Setup the architecture
    """
    n_embeddings = 20
    n_layers = 3
    n_hidden = 150

    def __init__(self, names, next_char_in_name):

        """
        Map each sequence of encoded names to a sequence of embeddings.
        """
        embedding_matrix = tf.Variable(tf.random_uniform([CharCodec.vocab_size, self.n_embeddings], -1.0, 1.0)) #(vocab_size, n_embeddings)
        embedded_names = tf.nn.embedding_lookup(embedding_matrix, names) #(?, seq_length, n_embeddings)

        """
        Next, we define one recurrent cell. The one that takes a vector of dimension n_embeddings as the input and
        generates a vector of size n_hidden as the output. This is for a single step. 
        """
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden) #(1, n_hidden)

        """
        Dynamic RNN helps us in repeating this LSTM cell as many times as needed. In our case, we need one LSTM 
        cell per character in our name. Thus, we feed in our embedded names vector, which has the
        dimensions (?, seq_length, n_embeddings). Note that the lstm_cell is a mapping from n_embeddings -> n_hidden.
        The dymanic_rnn function basically creates max_name_length of these cells, and emits the following two 
        vectors:
        - rnn_output_per_step (?, seq_length, n_hidden)
        - states (?, n_hidden)
        There is one output per lstm_cell, and there is final state.
        """
        rnn_output_per_step, states = tf.nn.dynamic_rnn(lstm_cell, inputs=embedded_names, dtype=tf.float32)

        """
        We feed the per step output to a simple dense layer that predicts the next character. There is a bit of an optimization
        at play here. The RNN we have has max_name_length nodes, each of which generates a vector that has is n_hidden dimensional.
        We want to feed each of these outputs to a Dense layer, and then generate a logits vector that's vocab_size dimensional.
        This logits vector is then fed to a softmax to find the winning character. To manage this operation, we first concatenate
        all the logits across the name. Thus, we get a single tensor that's (batch_size * max_name_length, n_hidden). This is then
        fed to a dense layer, to generate (batch_size * max_name_length, vocab_size). A reshape takes us back to 
        (batch_size, max_name_length, vocab_size)
        """
        rnn_output_per_step_stacked = tf.reshape(rnn_output_per_step, shape=[-1, self.n_hidden]) #(?, n_hidden)
        dense_layer_out_stacked = tf.layers.dense(rnn_output_per_step_stacked, CharCodec.vocab_size) #(?, vocab_size)
        predicted_next_char_logits_stacked = tf.reshape(dense_layer_out_stacked, shape=[-1, CharCodec.vocab_size]) #(?, vocab_size)


        self._loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=next_char_in_name,
                                                                           logits=predicted_next_char_logits_stacked))
        
        optimizer = tf.train.AdamOptimizer()
        self._train_op = optimizer.minimize(self.loss)
        # prediction

        predicted_next_char_logits = tf.reshape(dense_layer_out_stacked,
                                                shape=[-1, CharCodec.max_name_length,
                                                       CharCodec.vocab_size])  # (?, vocab_size)

        self._best_chars = tf.argmax(predicted_next_char_logits, axis=2)

    @property
    def prediction(self):
        return self._best_chars
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def loss(self):
        return self._loss
