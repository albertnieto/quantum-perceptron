# models/common.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset():
    """
    Loads a simple binary classification dataset that is linearly separable.
    With flip_y=0 and class_sep=2 this ensures that there exists at least 
    one weight vector that can perfectly classify all training examples.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=10,
        n_features=12,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,
        class_sep=2,
        random_state=42
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of predictions.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_true == y_pred)
