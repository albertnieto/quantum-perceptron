# models/online_quantum_perceptron.py

import pennylane as qml
from pennylane import numpy as np

def online_quantum_perceptron(X_train, y_train):
    """
    Online Quantum Perceptron updates weights using quantum subroutines.

    Args:
        X_train (array): Training features.
        y_train (array): Training labels.

    Returns:
        tuple: (Final weight vector, number of updates)
    """
    n_qubits = X_train.shape[1]
    dev = qml.device('default.qubit', wires=n_qubits)

    weights = np.zeros(n_qubits)
    update_count = 0

    @qml.qnode(dev)
    def quantum_inner_product(x_i, weights):
        """
        Quantum circuit to compute the inner product between x_i and weights.
        """
        for i in range(n_qubits):
            angle = (x_i[i] * weights[i]) * np.pi / 2
            qml.RY(angle, wires=i)
        # Measure expectation value of PauliZ on the first qubit
        return qml.expval(qml.PauliZ(0))

    for x_i, y_i in zip(X_train, y_train):
        # Compute quantum inner product
        prediction = quantum_inner_product(x_i, weights)
        pred_sign = 1 if prediction >= 0 else -1
        if pred_sign != y_i:
            # Quantum-inspired update (classical in this simplified example)
            weights += y_i * x_i
            update_count += 1

    return weights, update_count
