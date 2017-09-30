import abc
from gensim.models import KeyedVectors
import numpy
import fasttext
import more_itertools

import config
from common import util

class Vocabulary(object):
    def __init__(self, embedding: WordEmbedding, word_set: set):
        self.unk = '<unk>'
        self.id_to_word_dict = dict(enumerate(set(word_set), start=1))
        self.id_to_word_dict[0] = self.unk
        self.word_to_id_dict = util.reverse_dict(self.id_to_word_dict)
        self._embedding_matrix = [embedding[b] for a, b in sorted(self.id_to_word_dict.items(), key=lambda x:x[0])]

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


@util.disk_cache('word_vocabulary', config.cache_path)
def load_vocabulary(word_vector_name, text_list) -> Vocabulary:
    namd_embedding_dict = {"glove": GloveWordEmbedding()}
    word_set = more_itertools.collapse(text_list)
    return Vocabulary(namd_embedding_dict[word_vector_name], word_set)
