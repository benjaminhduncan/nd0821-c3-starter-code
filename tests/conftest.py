import pytest
import pandas as pd
from starter.train_model import load_data
from starter.ml.model import read_model, TrainedModel


@pytest.fixture
def data_fixture():
    # TODO: bring path out to a config
    data = pd.read_csv('data/census.csv')
    return data


@pytest.fixture
def training_data_fixture():
    # TODO: bring path out to a config
    data_path = 'data/census.csv'
    training_data = load_data(data_path)
    return training_data


# @pytest.fixture
# def trained_model_fixture():
#     # TODO: bring path out to a config
#     model_path = 'model/trained_model.joblib'
#     trained_model = read_model(model_path)
#     return trained_model
