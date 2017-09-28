import json
import os
import typing

import more_itertools
from nltk.tokenize import StanfordTokenizer
from sklearn.utils import shuffle

import config
from common import util
from common.util import parallel_map


## There is some bugs in the SQuAD._load_format_data function. The bug is fixed in the train set method and
## the validation_set_list propertyvalidation_set_list

class SQuAD(object):
    def __init__(self):
        self._train_data = self._load_format_data(False)
        self._validation_data = self._load_format_data(True)

    @staticmethod
    def _tokenizer(texts: typing.List[str]) -> typing.List[typing.List[str]]:
        """
        :param texts: a list of string to tokenize
        :return:
        """
        tokenizer = StanfordTokenizer()
        return tokenizer.tokenize_sents(texts)

    @staticmethod
    @util.disk_cache("SQuAD_data.pkl", config.cache_path)
    def _load_format_data(is_validation):
        if not is_validation:
            data_path = config.SQuAD_train_path
        else:
            data_path = config.SQuAD_dev_path

        with open(data_path, 'r') as f:
            data = json.load(f)
        if not is_validation:
            return SQuAD.preprocess_train_data(data)
        else:
            return SQuAD.preprocess_validation_data(data)

    @staticmethod
    def preprocess_validation_data(data):
        contexts = []
        quesitons = []
        ids = []
        print("total document number: {}".format(len(data['data'])))
        for i, document in enumerate(data['data']):
            for paragraphs in document['paragraphs']:
                for qa in paragraphs['qas']:
                    contexts.append(paragraphs['context'])
                    quesitons.append(qa['question'])
                    ids.append(qa['id'])
            if i % 10 == 0:
                print("finish the {}th document".format(i))
        core_number = 10
        partition = lambda x: [list(t) for t in more_itertools.divide(core_number, x)]
        contexts = parallel_map(core_number, SQuAD._tokenizer, contexts, partition, more_itertools.flatten)
        quesitons = parallel_map(core_number, SQuAD._tokenizer, quesitons, partition, more_itertools.flatten)
        return ids, contexts, quesitons, data

    @staticmethod
    def preprocess_train_data(data):
        contexts = []
        quesitons = []
        answer_texts = []
        answer_starts = []
        answer_ends = []
        print("total document number: {}".format(len(data['data'])))
        for i, document in enumerate(data['data']):
            for paragraphs in document['paragraphs']:
                for qa in paragraphs['qas']:
                    contexts.append(paragraphs['context'])
                    quesitons.append(qa['question'])
                    answer_texts.append(qa['answers'][0]['text'])
                    answer_starts.append(qa['answers'][0]['answer_start'])
                    answer_ends.append(qa['answers'][0]['answer_start'] + len(answer_texts[-1]))
            if i % 10 == 0:
                print("finish the {}th document".format(i))
        core_number = 10
        partition = lambda x: [list(t) for t in more_itertools.divide(core_number, x)]
        contexts = parallel_map(core_number, SQuAD._tokenizer, contexts, partition, more_itertools.flatten)
        quesitons = parallel_map(core_number, SQuAD._tokenizer, quesitons, partition, more_itertools.flatten)
        answer_texts = parallel_map(core_number, SQuAD._tokenizer, answer_texts, partition, more_itertools.flatten)
        return answer_ends, answer_starts, answer_texts, contexts, quesitons

    def train_set(self, epoches, batch_size=32):
        """
        It is a generator return a generator of the train data.
        Every time the generator will return a (paragraph(words_list), question, answer_text, answer_start) tuple
        """
        answer_ends, answer_starts, answer_texts, contexts, quesitons = self._train_data
        answer_ends = [start + len(tokens) for start, tokens in zip(answer_starts, answer_texts)]

        for _ in epoches:
            contexts, quesitons, answer_texts, answer_starts, answer_ends = shuffle(contexts, quesitons, answer_texts,
                                                                                    answer_starts, answer_ends)
            for i in range(0, len(contexts), batch_size):
                yield contexts[i:i + batch_size], \
                      quesitons[i:i + batch_size], \
                      answer_texts[i:i + batch_size], \
                      answer_starts[i:i + batch_size], \
                      answer_ends[i:i + batch_size]

    @property
    def validation_set_list(self):
        """
        It is will return a tuple (ids(List), paragraphs(List), questions(List), validation_data(raw json data))
        """
        return self._validation_data
