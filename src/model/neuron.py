# A neuron is a computational unit that receives one or more inputs, 
# applies weights to these inputs, sums them up, 
# and then passes the sum through an activation function to produce an output.

import numpy as np

class Neuron:
    """
    A class representing a single neuron in a neural network.

    Attributes:
        activation (function): Activation function to apply to the neuron's output.
        initialization (str): Initialization method for weights ('xavier' or 'he').
        learning_rate_schedule (bool): Whether to apply learning rate scheduling during training.
    """

    def __init__(self, activation, initialization='xavier', learning_rate_schedule=False):
        """
        Initialize the neuron with specified activation function and weight initialization method.

        Args:
            activation (str): Name of the activation function.
            initialization (str, optional): Initialization method for weights. Defaults to 'xavier'.
            learning_rate_schedule (bool, optional): Whether to apply learning rate scheduling during training. Defaults to False.
        """
        self.activation = self._get_activation(activation)
        self.weights = None
        self.bias = None
        self.initialization = initialization
        self.learning_rate_schedule = learning_rate_schedule

    def _get_activation(self, activation):
        """
        Get the activation function based on the provided name.

        Args:
            activation (str): Name of the activation function.

        Returns:
            function: Activation function corresponding to the provided name.

        Raises:
            ValueError: If the provided activation function name is not supported.
        """
        activation_funcs = {
            'linear':self.linear,
            'step': self.step,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu,
            'leaky_relu': self.leaky_relu,
            'softmax': self.softmax
        }
        if activation in activation_funcs:
            return activation_funcs[activation]
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

    def initialize(self, input_size):
        """
        Initialize the weights and bias of the neuron.

        Args:
            input_size (int): Size of the input features.
        """
        if self.initialization == 'xavier':
            self.weights = np.random.randn(input_size) / np.sqrt(input_size)
        elif self.initialization == 'he':
            self.weights = np.random.randn(input_size) * np.sqrt(2 / input_size)
        else:
            raise ValueError(f"Invalid initialization method: {self.initialization}")

        self.bias = np.random.randn()

    def predict(self, X):
        """
        Predict the output of the neuron for input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted outputs.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def fit(self, X, y, initial_learning_rate=0.01, n_iters=1000):
        """
        Train the neuron using gradient descent.

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target labels.
            initial_learning_rate (float, optional): Initial learning rate. Defaults to 0.01.
            n_iters (int, optional): Number of training iterations. Defaults to 1000.
        """
        n_samples, n_features = X.shape
        self.initialize(n_features)

        # Assuming binary classification
        if np.unique(y).size == 2:
            y_ = np.where(y > 0, 1, 0)
        else:
            y_ = y  # For other types of classification, leave y as it is

        learning_rate = initial_learning_rate
        for epoch in range(1, n_iters + 1):
            y_predicted = self.predict(X)
            errors = y_ - y_predicted

            self.weights += learning_rate * np.dot(X.T, errors)
            self.bias += learning_rate * np.sum(errors)

            if self.learning_rate_schedule:
                learning_rate = initial_learning_rate / np.sqrt(epoch)

    # Activation functions
    @staticmethod
    def linear(x):
        """
        Linear activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying linear function.
        """
        return x
    
    @staticmethod
    def step(x):
        """
        Step activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying step function.
        """
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying hyperbolic tangent function.
        """
        return np.tanh(x)

    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying ReLU function.
        """
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU activation function.

        Args:
            x (numpy.ndarray): Input data.
            alpha (float, optional): Slope of the negative part. Defaults to 0.01.

        Returns:
            numpy.ndarray: Output after applying Leaky ReLU function.
        """
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def softmax(x):
        """
        Softmax activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying softmax function.
        """
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
