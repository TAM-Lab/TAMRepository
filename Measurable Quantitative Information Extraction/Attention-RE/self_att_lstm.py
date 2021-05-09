import tensorflow as tf

# 忽略警告输出
import warnings

warnings.filterwarnings('ignore')


class AttLSTM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, num_heads, max_len, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout

        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # Word Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, input_keep_prob=self.rnn_dropout_keep_prob,
                                                    output_keep_prob=self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, input_keep_prob=self.rnn_dropout_keep_prob,
                                                    output_keep_prob=self.rnn_dropout_keep_prob)
            # rnn_outpus: [batch_size, max_length, hidden_num],
            # self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
            (self.rnn_outputs_fw, self.rnn_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                            cell_bw=bw_cell,
                                                                                            inputs=self.embedded_chars,
                                                                                            sequence_length=self._length(
                                                                                                self.input_text),
                                                                                            dtype=tf.float32)
            self.rnn_outputs = tf.concat([self.rnn_outputs_fw, self.rnn_outputs_bw], axis=-1)
            self.rnn_outputs = self.self_attention(self.rnn_outputs, hidden_size, num_heads)
        self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, 2 * hidden_size * max_len])
        W = tf.get_variable(name="W", shape=[2 * hidden_size * max_len, hidden_size], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=[hidden_size], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        self.rnn_outputs = tf.tanh(tf.nn.xw_plus_b(self.rnn_outputs, W, b))
        self.rnn_outputs = tf.nn.dropout(self.rnn_outputs, self.dropout_keep_prob)
        self.logits_W = tf.get_variable(name="logits_weight", shape=[hidden_size, num_classes], dtype=tf.float32)
        self.logits_b = tf.get_variable(name="logits_bias", shape=[num_classes], dtype=tf.float32)
        pred = tf.nn.xw_plus_b(self.rnn_outputs, self.logits_W, self.logits_b)
        self.predictions = tf.argmax(pred, 1, name="predictions")
        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

    def self_attention(self, keys, hidden_size, num_heads):
        Q = tf.layers.dense(tf.nn.relu(
            tf.layers.dense(keys, 2 * hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        ), 2 * hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.dense(tf.nn.relu(
            tf.layers.dense(keys, 2 * hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        ), 2 * hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.dense(tf.nn.relu(
            tf.layers.dense(keys, 2 * hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        ), 2 * hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # 分为多头
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # outputs += keys
        return outputs

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
