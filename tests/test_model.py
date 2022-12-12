import numpy
import pytest
from sklearn.ensemble._gb import GradientBoostingClassifier
from starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def test_train_model(training_data_fixture):
    model = train_model(training_data_fixture.X_train,
                        training_data_fixture.y_train)
    assert isinstance(model, GradientBoostingClassifier)
    return model


@pytest.fixture
def test_inference(test_train_model, training_data_fixture):
    preds = inference(test_train_model, training_data_fixture.X_test)
    assert isinstance(preds, numpy.ndarray)
    return preds


def test_compute_model_metrics(training_data_fixture, test_inference):
    precision, recall, fbeta = compute_model_metrics(
        training_data_fixture.y_test, test_inference)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
