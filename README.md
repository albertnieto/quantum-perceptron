# Quantum perceptron

This repository implements three different quantum perceptron models using Grover's algorithm with PennyLane. It also includes a classical perceptron for comparison. The models are designed to share common functions where applicable. Comprehensive tests are provided to ensure correctness, and a Jupyter notebook demonstrates executions, results, and comparisons.

## Repository structure

- `models/`: Contains implementations of classical and quantum perceptrons.
- `tests/`: Unit tests for each model and shared utilities.
- `notebooks/`: Jupyter notebook for executing and comparing models.
- `requirements.txt`: Dependencies required to run the code.

## Setup instructions

### 1. Clone the repository

```bash
git clone https://github.com/albertnieto/quantum-perceptron.git
cd quantum_perceptron
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run tests
```bash
pytest
```

### 5. Run the Jupyter notebook
```bash
jupyter notebook notebooks/quantum_perceptron_comparison.ipynb
```