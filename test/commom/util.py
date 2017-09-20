import unittest
from common import util

class Test(unittest.TestCase):
    def test_reverse_dict(self):
        d = {1: 3, 3: 4, 5: 7}
        t = util.reverse_dict(d)
        true_t = {3: 1, 4: 3, 7: 5}
        self.assertEqual(t, true_t, "reverse_dict error, function result {}, the target result {}".format(t, true_t))