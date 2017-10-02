import unittest
from embedding.character_embedding import load_character_vocabulary

class BiCharacterEmbeddingTest(unittest.TestCase):
    def setUp(self):
        text = [["abc", ["bca"]], ["ade", ["adt"]]]
        n_gram = 1
        embedding_shape = 300
        self.character_vocabulary = load_character_vocabulary("bigru", n_gram, embedding_shape, text)
        print("character_to_id_dict:{}".format(self.character_vocabulary.character_to_id_dict))
        print("id_to_character_dict:{}".format(self.character_vocabulary.id_to_character_dict))

    def test_parse_string(self):
        source_string = [["abe", "abt", "aet"], ["abc", "bac"]]
        self.character_vocabulary.parse_string(source_string)