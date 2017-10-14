import abc
from gensim.models import KeyedVectors
import numpy
import fasttext
import more_itertools
import tensorflow as tf
import numpy as np

import config
from common import util


class WordEmbedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


class GloveWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__()
        self.model = loadGloveModel(config.pretrained_glove_path)

    def __getitem__(self, item):
        if item in self.model:
            return self.model[item]
        else:
            return numpy.random.randn(*self.model['office'].shape)

# TODO: fasttext library has some error
# class FastTextWordEmbedding(WordEmbedding):
#     def __init__(self):
#         super().__init__()
#         self.model = fasttext.load_model(config.pretrained_fasttext_path)
#
#     def __getitem__(self, item):
#         pass


class Vocabulary(object):
    def __init__(self, embedding: WordEmbedding, word_set: set):
        self.unk = '<unk>'
        self.id_to_word_dict = dict(list(enumerate(set(word_set), start=1)))
        self.id_to_word_dict[0] = self.unk
        self.word_to_id_dict = util.reverse_dict(self.id_to_word_dict)
        self._embedding_matrix = np.array([embedding[b] for a, b in sorted(self.id_to_word_dict.items(), key=lambda x:x[0])])

    def word_to_id(self, word):
        if word in self.word_to_id_dict.keys():
            return self.word_to_id_dict[word]
        else:
            return 0

    def id_to_word(self, i):
        if i:
            return self.id_to_word_dict[i]
        else:
            return self.unk

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    def create_embedding_layer(self):
        with tf.variable_scope("word_embedding"):
            _tf_embedding = tf.Variable(name="embedding", initial_value=self._embedding_matrix,
                                             dtype=tf.float32, trainable=False)
        def embedding_layer(input_op):
            """
            :param input_op: a tensorflow tensor with shape [batch, max_length] and type tf.int32
            :return: a looked tensor with shape [batch, max_length, embedding_size]
            """
            output = tf.nn.embedding_lookup(_tf_embedding, input_op)
            print("word embedding:{}".format(output))
            return output

        return embedding_layer

    def parse_text(self, texts):
        """
        :param texts: a list of list of token
        :return:
        """
        max_text = max(map(lambda x:len(x), texts))
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        texts = [text+[0]*(max_text-len(text)) for text in texts]
        return texts


def load_vocabulary(word_vector_name, text_list) -> Vocabulary:
    namd_embedding_dict = {"glove": GloveWordEmbedding()}
    word_set = more_itertools.collapse(text_list)
    return Vocabulary(namd_embedding_dict[word_vector_name], word_set)
