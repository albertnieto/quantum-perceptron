# models/classical_perceptron.py

from sklearn.linear_model import Perceptron as SKPerceptron
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=1.0, max_iter=1000):
        self.model = SKPerceptron(
            eta0=learning_rate,
            max_iter=max_iter,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        self.convergence_steps = 0

    def train(self, X, y):
        self.model.fit(X, y)
        self.convergence_steps = self.model.n_iter_

    def predict(self, X):
        return self.model.predict(X)
