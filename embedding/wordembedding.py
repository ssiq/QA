import abc
from gensim.models import KeyedVectors
from collections import namedtuple

import config

WordDictionary = namedtuple("WordDictionary", ["embedding", ""])

class WordEmbedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def embedding_matrix(self, word_set):
        """
        :param word_set: a set of words
        :return:
        """
        return

    @abc.abstractmethod
    def __getitem__(self, item):
        pass


class GloveWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__()
        self.model = KeyedVectors.load(config.pretrained_glove_path)

    def __getitem__(self, item):
        return self.model[item]

    def embedding_matrix(self, word_set):
        pass