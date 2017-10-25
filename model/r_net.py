import tensorflow as tf
import numpy as np

from common import tf_util, util, rnn_cell, rnn_util

class RNet(object):
    def __init__(self,
                 word_embedding_layer,
                 charadter_embedding,
                 hidden_state_size,
                 rnn_layer_number,
                 learning_rate):
        self.passage_word_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="passage_word_input")
        self.passage_length = tf.placeholder(dtype=tf.int32, shape=(None,), name="passage_input_length")
        self.passage_character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None),
                                                      name="passage_character_input")
        self.passage_character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None),
                                                             name="passage_input_character_length")

        self.question_word_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="question_word_input")
        self.question_length = tf.placeholder(dtype=tf.int32, shape=(None,), name="question_length")
        self.question_character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None),
                                                       name="question_character_input")
        self.question_character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None),
                                                              name="question_character_input_length")

        self.answer_start_label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="answer_start")
        self.answer_end_label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="answer_end")

        self.word_embedding_layer = word_embedding_layer
        self.character_embedding_layer = charadter_embedding

        self.hidden_state_size = hidden_state_size
        self.rnn_layer_number = rnn_layer_number
        self.batch_size = tf_util.get_shape(self.question_word_input)[0]

        self.learning_rate = learning_rate
        self.global_variable = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)

        # tf_util.init_all_op(self)


    def _embedding(self, words, characters, character_length):
        return tf.concat((self.word_embedding_layer(words),
                          self.character_embedding_layer(characters, character_length)), axis=2)

    @tf_util.define_scope("passage_embedding_op")
    def passage_embedding_op(self):
        return self._embedding(self.passage_word_input, self.passage_character_input, self.passage_character_input_length)

    @tf_util.define_scope("question_embedding_op")
    def question_embedding_op(self):
        return self._embedding(self.question_word_input, self.question_character_input, self.question_character_input_length)

    def _gru_cell(self):
        return tf.nn.rnn_cell.GRUCell(self.hidden_state_size)

    def _multi_rnn_cell(self):
        return tf.nn.rnn_cell.MultiRNNCell([self._gru_cell() for _ in range(self.rnn_layer_number)],
                                           state_is_tuple=True)

    @tf_util.define_scope("passage_bi_rnn_op")
    def passage_bi_rnn_op(self):
        return rnn_util.concat_bi_rnn_output(rnn_util.bi_rnn(self._multi_rnn_cell, self.passage_embedding_op,
                                                             self.passage_length))

    @tf_util.define_scope("question_bi_rnn_op")
    def question_bi_rnn_op(self):
        return rnn_util.concat_bi_rnn_output(rnn_util.bi_rnn(self._multi_rnn_cell,
                                                             self.question_embedding_op,
                                                             self.question_length))

    @tf_util.define_scope("question_aware_passage_op")
    def question_aware_passage_op(self):
        return rnn_util.concat_bi_rnn_output(rnn_util.bi_rnn(
            lambda: rnn_cell.GatedAttentionWrapper(self._multi_rnn_cell(), self.question_bi_rnn_op,
                                                   self.question_length,
                                                   self.hidden_state_size, ), self.passage_bi_rnn_op,
            self.passage_length))

    @tf_util.define_scope("self_matching_attention_passage")
    def self_matching_passage_op(self):
        return rnn_util.bi_rnn(
            lambda: rnn_cell.SelfMatchAttentionWrapper(self._multi_rnn_cell(),
                                                       self.question_aware_passage_op,
                                                       self.question_length,
                                                       self.hidden_state_size),
            self.question_aware_passage_op,
            self.question_length
        )[0]

    @tf_util.define_scope("question_vector_op")
    def question_vector_op(self):
        Vr = tf.get_variable("question_Vr",
                             shape=(1, self.hidden_state_size),
                             dtype=tf.float32)
        return rnn_util.soft_attention_reduce_sum(self.question_bi_rnn_op,
                                                         Vr,
                                                         self.hidden_state_size,
                                                         self.question_length)[0]

    @tf_util.define_scope("answer")
    def answer_logit_op(self):
        with tf.variable_scope("soft_attention"):
            a0 = rnn_util.soft_attention_logit(self.hidden_state_size,
                                                      self.question_vector_op,
                                                      self.self_matching_passage_op,
                                                      self.passage_length)
        a0_softmax = tf.expand_dims(tf.nn.softmax(a0), dim=2)
        c0 = sum(a0_softmax*m for m in self.self_matching_passage_op)
        cell = self._gru_cell()
        h1 = cell(c0, self.question_vector_op)
        with tf.variable_scope("soft_attention", reuse=True):
            a1 = rnn_util.soft_attention_logit(self.hidden_state_size,
                                                      h1,
                                                      self.self_matching_passage_op,
                                                      self.passage_length)
        return a0, a1

    @tf_util.define_scope("loss")
    def loss_op(self):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_start_label,
                                                              logits=self.answer_logit_op[0]) + \
               tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_end_label,
                                                              logits=self.answer_logit_op[1])

    @tf_util.define_scope("answer_predict")
    def answer_predict_op(self):
        return tf.argmax(self.answer_logit_op[0], axis=1), tf.argmax(self.answer_logit_op[1], axis=1)

    @tf_util.define_scope("train")
    def train_op(self):
        optimizar = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate,
                                               epsilon=1e-6)
        return tf_util.minimize_and_clip(optimizar,
                                         self.loss_op,
                                         tf.trainable_variables(),
                                         global_step=self.global_variable)