# tests/test_common.py

import unittest
import numpy as np
from models import common

class TestCommon(unittest.TestCase):
    def test_accuracy(self):
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([1, 1, -1, -1])
        acc = common.accuracy(y_true, y_pred)
        self.assertEqual(acc, 0.5)

    def test_load_dataset(self):
        X_train, X_test, y_train, y_test = common.load_dataset()
        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(X_test), 4)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 4)

if __name__ == '__main__':
    unittest.main()
