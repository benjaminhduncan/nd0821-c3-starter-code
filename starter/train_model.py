# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from typing import Any, Union
import pandas as pd
from dataclasses import dataclass
from joblib import dump


@dataclass
class TrainedModel:
    # TODO: write deocstring
    model: Any
    encoder: Any
    lb: Any
    precision: Any
    recall: Any
    fbeta: Any


@dataclass
class TrainingData:
    # TODO: write deocstring
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    encoder: Any
    lb: Any


def load_data(data_path: str):
    """
    Loads Census data, splits the sets, pre-processes and returns it.

    Inputs
    ------
    data_path
        A string representation of the absolute path of the data in csv form.
        Optionaly, a DataFrame can be provided directly.
    Returns
    -------
    training_data
        TrainingData object containing training data, test data, the encoder and lb.
    """

    data = pd.read_csv(data_path)
    data = data.dropna()

    # Split data set into train and test
    train, test = train_test_split(
        data, test_size=0.20, stratify=data[['race', 'sex']])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Pre-process the training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Pre-proces the test data
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    training_data = TrainingData(
        X_train,
        y_train,
        X_test,
        y_test,
        encoder,
        lb
    )

    return training_data


def validate_model(training_data: TrainingData, output_path: str):
    """
    Trains the model, computes metrics and returns the trained model.

    Inputs
    ------
    training_data
        TrainingData object containing training data, test data, the encoder and lb.
    output_path
        Output path for the model object.
    Returns
    -------
    trained_model
        TrainedModel object containing model, metrics, encoder and lb.
    """

    # Train the model
    model = train_model(training_data.X_train, training_data.y_train)

    # Perform inference on test set
    preds = inference(model, training_data.X_test)

    # Compute performance metrics
    precision, recall, fbeta = compute_model_metrics(
        training_data.y_test, preds)

    # Build and save trained model data object
    trained_model = TrainedModel(
        model,
        training_data.encoder,
        training_data.lb,
        precision,
        recall,
        fbeta
    )
    dump(trained_model, output_path)


if __name__ == "__main__":
    # TODO: convert config to hydra
    data_path = 'data/census.csv'
    output_path = 'model/trained_model.joblib'
    training_data = load_data(data_path)
    trained_model = validate_model(training_data, output_path)
