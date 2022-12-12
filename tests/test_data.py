"""
Module for testing the ml data functions
"""

import numpy
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelBinarizer

from ml.data import process_data


def test_process_data(data_fixture):
    """
    Function used to test the data pre-processing
    """
    # TODO: bring categoricals out to config
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

    X, y, encoder, lb = process_data(
        data_fixture,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=None,
        lb=None
    )

    assert isinstance(X, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
