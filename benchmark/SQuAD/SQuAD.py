from . import evaluate
import config
import json
import typing
from sklearn.utils import shuffle
from nltk.tokenize import StanfordTokenizer
import util
import os

class SQuAD(object):
    def __abs__(self):
        self._train_data = self._load_format_data(False)
        self._validation_data = self._load_format_data(True)
        self.tokenizer = StanfordTokenizer()

    def _tokenizer(self, text: str) -> typing.List[str]:
        """
        :param text: a string to tokenize
        :return:
        """
        return self.tokenizer.tokenize(text)

    @util.disk_cache("SQuAD_data.pkl", os.path.join("question answer", "dataset", "cache"))
    def _load_format_data(self, is_validation):
        if not is_validation:
            data_path = config.SQuAD_train_path
        else:
            data_path = config.SQuAD_dev_path

        with open(data_path, 'r') as f:
            data = json.load(f)
        contexts = []
        quesitons = []
        answer_texts = []
        answer_starts = []
        answer_ends = []
        for document in data['data']:
            for paragraphs in document['paragraphs']:
                for qa in paragraphs['qas']:
                    contexts.append(self._tokenizer(paragraphs['context']))
                    quesitons.append(self._tokenizer(qa['question']))
                    answer_texts.append(self._tokenizer(qa['answers'][0]['text']))
                    answer_starts.append(qa['answers'][0]['answer_start'])
                    answer_ends.append(qa['answers'][0]['answer_start'] + len(answer_texts[-1]))
        return answer_ends, answer_starts, answer_texts, contexts, quesitons

    def train_set(self, epoches, batch_size=32):
        """
        It is a generator return a generator of the train data.
        Every time the generator will return a (paragraph(words_list), question, answer_text, answer_start) tuple
        """
        answer_ends, answer_starts, answer_texts, contexts, quesitons = self._train_data

        for _ in epoches:
            contexts, quesitons, answer_texts, answer_starts, answer_ends = shuffle(contexts, quesitons, answer_texts, answer_starts, answer_ends)
            for i in range(0, len(contexts), batch_size):
                yield contexts[i:i+batch_size], \
                      quesitons[i:i+batch_size], \
                      answer_texts[i:i+batch_size], \
                      answer_starts[i:i+batch_size], \
                      answer_ends[i:i+batch_size]

    @property
    def validation_set_list(self):
        """
        It is will return a list of tuple (id, paragraph, question)
        """
        return self._validation_data
