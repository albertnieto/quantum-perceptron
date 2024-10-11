# tests/test_hybrid_quantum_perceptron.py

import unittest
import numpy as np
from models import hybrid_quantum_perceptron

class TestHybridQuantumPerceptron(unittest.TestCase):
    def test_hybrid_quantum_perceptron(self):
        X_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1],
                            [2, 2], [2, -2], [-2, 2], [-2, -2]])
        y_train = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        weights, update_count = hybrid_quantum_perceptron.hybrid_quantum_perceptron(X_train, y_train)
        predictions = np.sign(np.dot(X_train, weights))
        # Check if all predictions are correct
        self.assertTrue(np.all(predictions == y_train))
        # Check if updates were performed
        self.assertGreater(update_count, 0)

if __name__ == '__main__':
    unittest.main()
