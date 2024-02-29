import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    """
    Test train model
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    assert isinstance(model, BaseEstimator) and isinstance(
        model, ClassifierMixin)


def test_compute_model_metrics():
    """
    Test compute model metrics
    """
    y_true, y_preds = [1, 1, 0], [0, 1, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference():
    """
    Test inference
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    y_preds = inference(model, X)
    assert y.shape == y_preds.shape