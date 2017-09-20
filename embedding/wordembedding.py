import abc
from gensim.models import KeyedVectors

import config
from common import util

class Vocabulary(object):
    def __init__(self, embedding: WordEmbedding, word_set: set):
        self.unk = '<unk>'
        self.id_to_word_dict = dict(enumerate(set(word_set), start=1))
        self.id_to_word_dict[0] = self.unk
        self.word_to_id_dict = util.reverse_dict(self.id_to_word_dict)
        self.embedding = embedding

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

    def embedding_matrix(self):
        return [self.embedding[b] for a, b in sorted(self.id_to_word_dict.items(), key=lambda x:x[0])]

class WordEmbedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass


class GloveWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__()
        self.model = KeyedVectors.load(config.pretrained_glove_path)

    def __getitem__(self, item):
        return self.model[item]