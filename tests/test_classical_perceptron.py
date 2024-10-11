# tests/test_classical_perceptron.py

import unittest
import numpy as np
from models import classical_perceptron

class TestClassicalPerceptron(unittest.TestCase):
    def test_training_and_prediction(self):
        perceptron = classical_perceptron.Perceptron()
        X_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1],
                            [2, 2], [2, -2], [-2, 2], [-2, -2]])
        y_train = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        perceptron.train(X_train, y_train)
        X_test = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]])
        y_pred = perceptron.predict(X_test)
        self.assertEqual(len(y_pred), 4)
        # Since the dataset is linearly separable, expect high accuracy
        expected = np.array([1, -1, -1, 1])
        accuracy = np.mean(y_pred == expected)
        self.assertGreaterEqual(accuracy, 0.75)

if __name__ == '__main__':
    unittest.main()
