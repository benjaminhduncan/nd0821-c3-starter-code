"""
Module for testing the train model module functionality.
"""

from ml.train_model import load_data, validate_model
from ml.model import TrainedModel
from ml.data import TrainingData


def test_load_data():
    """
    Function to test loading the training data
    """
    # TODO: bring out to config
    data_path = 'data/census.csv'
    split_data_path = 'data/split_data.pkl'
    traing_data = load_data(data_path, split_data_path)
    assert isinstance(traing_data, TrainingData)


def test_validate_model(training_data_fixture):
    """
    Function to test the return of the trained model
    """
    trained_model = validate_model(training_data_fixture)
    assert isinstance(trained_model, TrainedModel)
