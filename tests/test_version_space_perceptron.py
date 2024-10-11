# tests/test_version_space_perceptron.py

import unittest
import numpy as np
from models import version_space_perceptron

class TestVersionSpaceQuantumPerceptron(unittest.TestCase):
    def test_version_space_quantum_perceptron(self):
        X_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1],
                            [2, 2], [2, -2], [-2, 2], [-2, -2]])
        y_train = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        weights, iterations = version_space_perceptron.version_space_quantum_perceptron(X_train, y_train)
        predictions = np.sign(np.dot(X_train, weights))
        # Check if all predictions are correct
        self.assertTrue(np.all(predictions == y_train))
        # Check if iterations are reasonable
        self.assertGreater(iterations, 0)

if __name__ == '__main__':
    unittest.main()
