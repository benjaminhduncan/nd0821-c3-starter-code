# Script to train machine learning model.

from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import pandas as pd

from joblib import dump

# Add the necessary imports for the starter code.

data = pd.read_csv('data/census.csv')
data = data.dropna()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, stratify=data[['race', 'sex']])

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

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# Train and save a model.
model = train_model(X_train, y_train)
print(type(model))
preds = inference(model, X_test)
print(f"Preds: {preds}")
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Model Precision: {precision}\nModel Recall: {recall}\nModel fbeta: {fbeta}")
dump(model, 'model/model.joblib') 