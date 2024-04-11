# It takes multiple binary inputs, applies weights to them, sums them up, 
# and then passes the sum through an activation function, 
# typically a step function, to produce a binary output.
#
# Perceptrons are typically used for binary classification tasks and can only learn linear decision boundaries.

class Perceptron:
    """
    A simple perceptron classifier.

    Attributes:
        learning_rate (float): The learning rate for training.
        n_iters (int): The number of iterations for training.
        neuron (Neuron): The neuron instance used for classification.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initialize the perceptron.

        Args:
            learning_rate (float, optional): The learning rate for training. Defaults to 0.01.
            n_iters (int, optional): The number of iterations for training. Defaults to 1000.
        """
        self.neuron = Neuron(activation="step")
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        """
        Fit the perceptron to the training data.

        Args:
            X (numpy.ndarray): The input features.
            y (numpy.ndarray): The target labels.
        """
        self.neuron.fit(X, y, self.learning_rate, self.n_iters)

    def predict(self, X):
        """
        Predict the labels for input data.

        Args:
            X (numpy.ndarray): The input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        return self.neuron.predict(X)
    
    def get_weights(self):
        """
        Get the weights of the perceptron.

        Returns:
            numpy.ndarray: Weights of the perceptron.
        """
        return self.neuron.weights

    def get_bias(self):
        """
        Get the bias of the perceptron.

        Returns:
            float: Bias of the perceptron.
        """
        return self.neuron.bias
