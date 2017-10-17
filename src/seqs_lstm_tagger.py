import os
import numpy as np
import tensorflow as tf


class RNNModel(object):

    def __init__(self,
                 max_seq_length,
                 nb_tags,
                 rnn_hidden_size,
                 embedding_size,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
                 max_grad_norm=5.0,
                 nonlin=tf.tanh,
                 session=tf.Session(),):
        self._max_seq_length = max_seq_length
        self._num_tags = nb_tags
        self._rnn_hidden_size = rnn_hidden_size
        self._embedding_size = embedding_size
        self._initializer = initializer
        self._optimizer = optimizer
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._sess = session
        self._name = "LSTM-Tagger"

        self._build_inputs()

        # convert to one-hot labels for cross entropy
        self._labels_one_hot = tf.one_hot(self._labels, depth=self._num_tags)
        # [None, max_seq_length, nb_tags]

        self._seqs_length = self._sent_len(self._seqs)
        # [None]

        logits = self._inference(self._seqs, self._seqs_length)
        loss = self._loss(logits, self._labels_one_hot, self._seqs_length)
        train_op = self._optimize(loss)

        predict_op = tf.argmax(input=logits, dimension=2, name="predict_op")
        predict_dist_op = tf.nn.softmax(
            logits=logits, dim=-1, name="predict_dist_op")

        self.loss_op = loss
        self.train_op = train_op
        self.predict_op = predict_op
        self.predict_dist_op = predict_dist_op

        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        self._saver = tf.train.Saver()

    def _build_inputs(self):
        self._seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._max_seq_length, self._embedding_size],
            name="seqs"
        )  # 0: padding, 1: unk
        self._labels = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self._max_seq_length],
            name="labels"
        )  # paddings should be -1

    def _sent_len(self, sentence):
        '''
        Args:
            sentence: [None, max_seq_length, embedding_size]

        Returns:
            length: [None], the actual length of each sentence in the batch
        '''
        #used = tf.sign(tf.abs(sentence))
        used = tf.sign(tf.abs(tf.reduce_sum(sentence, 1)))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(x=length, dtype=tf.int32)
        return length

    def _tensor_dot(self, A, B):
        '''
        Args:
            A: [None, a, b]
            B: [b, c]

        Returns:
            AB: [None, a, c], A * B
        '''
        batch_size = tf.shape(A)[0]
        A_shape = A.get_shape().as_list()
        B_shape = B.get_shape().as_list()
        A_reshaped = tf.reshape(A, shape=[batch_size * A_shape[1], A_shape[2]])
        AB = tf.matmul(A_reshaped, B)
        return tf.reshape(AB, shape=[batch_size, A_shape[1], B_shape[1]])

    def _inference(self, seqs, seqs_length):
        '''
        Args:
            seqs: [None, max_seq_length]

        Returns:
            rnn2tags: [None, max_seq_length, num_tags]
        '''
        with tf.variable_scope(self._name):
            # [None, max_seq_length, embedding_size]
            seqs_emb = self._seqs
            lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(
                num_units=self._rnn_hidden_size)
            rnn_output_fw, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell_fw,
                inputs=seqs_emb,
                sequence_length=seqs_length,
                dtype=tf.float32,
            )
            # rnn_output_fw: [None, max_seq_length, rnn_hidden_size]

            rnn2tags_W_fw = tf.get_variable(
                name="rnn2tags_W_fw",
                shape=[self._rnn_hidden_size, self._num_tags],
                dtype=tf.float32,
                initializer=self._initializer,
                trainable=True,
            )
            rnn2tags_b = tf.get_variable(
                name="rnn2tags_b",
                shape=[1, 1, self._num_tags],
                dtype=tf.float32,
                initializer=self._initializer,
                trainable=True,
            )
            rnn2tags = self._nonlin(
                self._tensor_dot(rnn_output_fw, rnn2tags_W_fw)
                + rnn2tags_b,
                name="rnn2tags"
            )
            # [None, max_seq_length, num_tags]

        return rnn2tags

    def _loss(self, predictions, labels, seqs_length):
        '''
        Args:
            predictions: [None, max_seq_length, nb_tags]
            labels: [None, max_seq_length, nb_tags]

        Returns:
            loss: []
        '''
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=predictions,
            labels=tf.cast(x=labels, dtype=tf.float32),
            name="cross_entropy_without_mask",
        )
        # [None, max_seq_length]
        cross_entropy_mask = tf.sequence_mask(
            lengths=seqs_length,
            maxlen=self._max_seq_length,
            dtype=tf.float32,
        )
        # [None, max_seq_length]
        cross_entropy = tf.multiply(
            x=cross_entropy,
            y=cross_entropy_mask,
            name="cross_entropy_after_mask",
        )
        # [None, mas_sentence_length]
        cross_entropy = tf.reduce_sum(
            input_tensor=cross_entropy,
            reduction_indices=1,
            name="cross_entropy_sum_sentence",
        )
        # [None]
        cross_entropy = tf.div(
            x=cross_entropy,
            y=tf.cast(x=seqs_length, dtype=tf.float32),
            name="cross_entropy_normalized_by_sent_len",
        )
        # [None]
        loss = tf.reduce_mean(input_tensor=cross_entropy, name="loss")
        return loss

    def _optimize(self, loss):
        '''
        Args:
            loss: loss

        Returns:
            train_op: train op
        '''
        grads_and_vars = self._optimizer.compute_gradients(loss)
        grads_and_vars = [
            (tf.clip_by_norm(g, self._max_grad_norm), v)
            for g, v in grads_and_vars
        ]
        train_op = self._optimizer.apply_gradients(
            grads_and_vars,
            # global_step=self._global_step,
            name="train_op",
        )
        return train_op

    def _get_mini_batch_start_end(self, n_train, batch_size=None):
        '''
        Args:
            n_train: int, number of training instances
            batch_size: int (or None if full batch)

        Returns:
            batches: list of tuples of (start, end) of each mini batch
        '''
        mini_batch_size = n_train if batch_size is None else batch_size
        batches = zip(
            range(0, n_train, mini_batch_size),
            list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
        )
        return batches

    def fit(self, seqs, labels, batch_size=None):
        '''
        Args:
            seqs: [None, max_seq_length]
            labels: [None, max_seq_length]
            batch_size: int (or None if full batch)

        Returns:
            total_loss: float, mean total loss
        '''
        n_train = len(seqs)
        batches = self._get_mini_batch_start_end(n_train, batch_size)
        total_loss = 0.
        for start, end in batches:
            feed_dict = {
                self._seqs: seqs[start:end],
                self._labels: labels[start:end],
            }
            loss, _ = self._sess.run(
                [self.loss_op, self.train_op], feed_dict=feed_dict)
            total_loss += (loss * (end - start))
        return total_loss / n_train

    def predict(self, seqs, batch_size=None, return_dist=False):
        '''
        Args:
            seqs: [None, max_seq_length]
            batch_size: int (or None if full batch)
            return_dist: boolean, whether to return the predicted distribution

        Returns:
            predictions: list of predictions/distributions of actual sentence length
        '''
        n_train = len(seqs)
        batches = self._get_mini_batch_start_end(n_train, batch_size)
        predictions = []
        for start, end in batches:
            feed_dict = {
                self._seqs: seqs[start:end],
            }
            sent_len, pred = self._sess.run(
                [self._seqs_length,
                    self.predict_dist_op if return_dist else self.predict_op],
                feed_dict=feed_dict
            )
            for l, p in zip(sent_len, pred):
                predictions.append(p[:l])
        return predictions

    def save(self, checkpoint_prefix):
        return self._saver.save(self._sess, checkpoint_prefix)

    def restore(self, file_dir):
        saver = tf.train.import_meta_graph(
            os.path.join(file_dir, 'model.meta'))
        saver.restore(self._sess, tf.train.latest_checkpoint(file_dir))


# simple test for the code
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size [32]")
tf.app.flags.DEFINE_integer("rnn_hidden_size", 100, "RNN hidden size [100]")
tf.app.flags.DEFINE_integer("embedding_size", 1, "Embedding size [1]")
tf.app.flags.DEFINE_integer("epochs", 100, "Number of epochs [100]")
tf.app.flags.DEFINE_integer("random_seed", None, "Random seed [None]")
#tf.app.flags.DEFINE_string("opt", "sgd", "Optimizer [sgd]")
tf.app.flags.DEFINE_integer("lr_decay_period", 25,
                            "Learning rate decay period [25]")
tf.app.flags.DEFINE_float("lr_decay_ratio", 0.5,
                          "Learning rate decay ratio [0.5]")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
tf.app.flags.DEFINE_integer("evaluation_interval",
                            1, "Evaluation interval [1]")
tf.app.flags.DEFINE_boolean("case_folding", False, "Do case folding [False]")
tf.app.flags.DEFINE_string("save_model", None, "Save path")

FLAGS = tf.app.flags.FLAGS

max_sent_len = 10
nb_tags = 3
train_X = np.random.uniform(0, 1, size=(10, 10, 1))
train_Y = np.random.randint(3, size=(10, 10))

#print train_X, train_Y

optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

with tf.Session() as sess:
    tagger = RNNModel(
        max_seq_length=max_sent_len,
        nb_tags=nb_tags,
        rnn_hidden_size=FLAGS.rnn_hidden_size,
        embedding_size=FLAGS.embedding_size,
        optimizer=optimizer,
        session=sess,
    )

    i_loss = 0.0
    for i in range(10000):
        i_loss = tagger.fit(seqs=train_X, labels=train_Y, batch_size=None)
        print i_loss
