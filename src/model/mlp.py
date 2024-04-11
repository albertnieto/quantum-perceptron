class MLP:
    def __init__(self, input_neurons=1, output_neurons=1, inner_layers=[0], learning_rate=0.01, n_iters=1000):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.inner_layers = inner_layers
        self.neurons = self.create_network()
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        if len(inner_layers) < 0:
            raise ValueError("At least two layers are needed for an MLP.")
        
    def create_network(self):
        """
        Create the neural network with the specified architecture.

        Returns:
            list: List of neuron layers.
        """
        neurons = []

        # Input layer
        input_layer = [Neuron() for _ in range(self.input_neurons)]
        neurons.append(input_layer)

        # Inner layers
        for num_neurons in self.inner_layers:
            inner_layer = [Neuron() for _ in range(num_neurons)]
            neurons.append(inner_layer)

        # Output layer
        output_layer = [Neuron() for _ in range(self.output_neurons)]
        neurons.append(output_layer)

        return neurons

    def fit(self, X, y):
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                predicted = self.predict(x_i)
                errors = [y[idx] - predicted[-1]]
                for layer_idx in reversed(range(len(self.layers) - 1)):
                    errors.append(np.dot(errors[-1], self.neurons[layer_idx].weights.T))
                errors.reverse()
                for layer_idx, neuron in enumerate(self.neurons):
                    neuron.fit(X[idx], errors[layer_idx], self.learning_rate)

    def predict(self, X):
        outputs = [X]
        output = X
        for neuron in self.neurons:
            output = neuron.predict(output)
            outputs.append(output)
        return outputs

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def get_biases(self):
        return [neuron.bias for neuron in self.neurons]
