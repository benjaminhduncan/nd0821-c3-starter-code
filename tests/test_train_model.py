from starter.train_model import load_data, validate_model
from starter.ml.data import TrainingData
import tempfile
from pathlib import Path


def test_load_data():
    # TODO: bring out to config
    data_path = 'data/census.csv'
    traing_data = load_data(data_path)
    assert isinstance(traing_data, TrainingData)


def test_validate_model(training_data_fixture):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = Path(tmpdirname) / 'tmp_model.joblib'
        validate_model(training_data_fixture, model_path)
        assert model_path.exists()
