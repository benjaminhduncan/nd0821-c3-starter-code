"""
This module is used for testing the module functions
"""

import numpy
import pytest
from sklearn.ensemble._gb import GradientBoostingClassifier
from ml.model import train_model, inference, compute_model_metrics, dump_model, read_model


@pytest.fixture
def test_train_model(training_data_fixture):
    """
    Function to test training the model
    """
    model = train_model(training_data_fixture.X_train,
                        training_data_fixture.y_train)
    assert isinstance(model, GradientBoostingClassifier)
    return model


@pytest.fixture
def test_inference(test_train_model, training_data_fixture):
    """
    Function to test performing inference with the model
    """
    preds = inference(test_train_model, training_data_fixture.X_test)
    assert isinstance(preds, numpy.ndarray)
    return preds


def test_compute_model_metrics(training_data_fixture, test_inference):
    """
    Function to test computing the metrics for the model
    """
    precision, recall, fbeta = compute_model_metrics(
        training_data_fixture.y_test, test_inference)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_dump_model(test_train_model):
    pass


def test_read_model():
    pass
