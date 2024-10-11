# models/version_space_perceptron.py

import pennylane as qml
from pennylane import numpy as np

def version_space_quantum_perceptron(X_train, y_train):
    """
    Uses Grover's algorithm to find a weight vector that correctly classifies all training examples.

    Args:
        X_train (array): Training features.
        y_train (array): Training labels.

    Returns:
        tuple: (Weight vector, number of iterations)
    """
    n_qubits = X_train.shape[1]  # Number of qubits corresponds to number of features
    N = 2 ** n_qubits  # Total number of possible weight vectors
    iterations = int(np.pi / 4 * np.sqrt(N))  # Number of Grover iterations
    if iterations < 1:
        iterations = 1

    # Define the device
    dev = qml.device('default.qubit', wires=n_qubits)

    # Define the oracle
    def oracle():
        """
        Oracle that flips the phase of the state |11...1> if it correctly classifies all training examples.
        """
        # Implementing multi-controlled Z gate
        control_wires = list(range(n_qubits - 1))
        target_wire = n_qubits - 1

        # MultiControlledX to flip target qubit when all control qubits are |1>
        qml.MultiControlledX(wires=control_wires + [target_wire], control_values='1' * (n_qubits -1))
        # Apply Z gate to target qubit
        qml.PauliZ(wires=target_wire)
        # Undo MultiControlledX
        qml.MultiControlledX(wires=control_wires + [target_wire], control_values='1' * (n_qubits -1))

    # Define the diffusion operator
    def diffusion():
        # Apply Hadamard and Pauli-X gates
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
            qml.PauliX(wires=wire)

        # Apply multi-controlled Z gate
        control_wires = list(range(n_qubits -1))
        target_wire = n_qubits -1
        qml.MultiControlledX(wires=control_wires + [target_wire], control_values='1' * (n_qubits -1))
        qml.PauliZ(wires=target_wire)
        qml.MultiControlledX(wires=control_wires + [target_wire], control_values='1' * (n_qubits -1))

        # Apply Pauli-X and Hadamard gates
        for wire in range(n_qubits):
            qml.PauliX(wires=wire)
            qml.Hadamard(wires=wire)

    @qml.qnode(dev)
    def grover_circuit():
        # Initialize in uniform superposition
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)

        # Apply Grover iterations
        for _ in range(iterations):
            oracle()
            diffusion()

        return qml.probs(wires=range(n_qubits))

    # Run the circuit
    probabilities = grover_circuit()
    print("Probabilities after Grover's iterations:", probabilities)

    # Find the state with the highest probability
    max_state = np.argmax(probabilities)
    weights_binary = format(max_state, f'0{n_qubits}b')
    weights = np.array([1 if bit == '1' else -1 for bit in weights_binary])

    # Check if weights classify all training examples correctly
    predictions = np.sign(np.dot(X_train, weights))
    correct = np.all(predictions == y_train)
    if correct:
        print(f"Found a valid weight vector: {weights}")
    else:
        print(f"Did not find a valid weight vector. Best weights: {weights}")
        print(f"Classification on training data: {predictions}")
        print(f"Expected labels: {y_train}")

    return weights, iterations
