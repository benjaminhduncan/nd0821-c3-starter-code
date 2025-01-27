from main import app
from fastapi.testclient import TestClient
import pytest
import pandas as pd
from ml.train_model import load_data


@pytest.fixture
def data_fixture():
    # TODO: bring path out to a config
    data = pd.read_csv('data/census.csv')
    return data


@pytest.fixture
def training_data_fixture():
    # TODO: bring path out to a config
    data_path = 'data/census.csv'
    split_data_path = 'data/split_data.pkl'
    training_data = load_data(data_path, split_data_path)
    return training_data


# @pytest.fixture
# def trained_model_fixture():
#     # TODO: bring path out to a config
#     model_path = 'model/trained_model.joblib'
#     trained_model = read_model(model_path)
#     return trained_model


@pytest.fixture
def test_client_fixture():
    client = TestClient(app)
    return client
