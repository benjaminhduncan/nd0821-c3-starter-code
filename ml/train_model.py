"""
Script to train machine learning model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data, TrainingData
from ml.model import train_model, compute_model_metrics, inference, TrainedModel, dump_model


def load_data(data_pth: str) -> TrainingData:
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

    data = pd.read_csv(data_pth)
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
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
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


def validate_model(training_data: TrainingData) -> TrainedModel:
    """
    Trains the model, computes metrics and returns the trained model.

    Inputs
    ------
    training_data
        TrainingData object containing training data, test data, the encoder and lb.
    Returns
    -------
    trained_model
        TrainingModel object containing trained model and results.
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

    return trained_model


def main(data_path, output_path, training_data_path):
    """
    Main function for the train_model module.
    """
    # TODO: add a docstring
    training_data = load_data(data_path)
    dump_model(training_data, training_data_path)
    trained_model = validate_model(training_data)
    dump_model(trained_model, output_path)


if __name__ == "__main__":
    # TODO: convert config to hydra
    CENSUS_DATA_PATH = 'data/census.csv'
    TRAINING_DATA_PATH = 'data/training_data.pkl'
    MODEL_OUTPUT_PATH = 'model/trained_model.pkl'
    main(CENSUS_DATA_PATH, MODEL_OUTPUT_PATH, TRAINING_DATA_PATH)
