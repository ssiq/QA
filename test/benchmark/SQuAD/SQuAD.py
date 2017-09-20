from benchmark.SQuAD import SQuAD
from benchmark.SQuAD import evaluate_pair_list, evaluate
import unittest

class Test(unittest.TestCase):

    def parse_target_and_prediction(self, target_list, prediction_list):
        predictions = dict(enumerate(prediction_list))
        qas = list(map(lambda x: {'answers': [{'text': x[1]}], 'id': x[0]}, enumerate(target_list)))
        dataset = [{'paragraphs':[{'qas':qas}]}]
        return dataset, predictions


    def test_evaluate_pair_list(self):
        target_list = ["I will miss", "I will miss", "Hello World"]
        predict_list = ["i   will miss!", "abc ed", "hello world"]

        self.assertEqual(evaluate(*self.parse_target_and_prediction(target_list, predict_list)),
                         evaluate_pair_list(target_list, predict_list))

    def test_SQuAD(self):
        squad = SQuAD()
        print(len(squad.validation_set_list))
