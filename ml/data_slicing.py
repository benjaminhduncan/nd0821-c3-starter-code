"""
Script to perform inference on categorical data slices.
"""
import pandas as pd
from ml.model import read_model, inference, compute_model_metrics
from ml.data import process_data


def predict_slice(data, feature, category, trained_model, cat_features):
    """ Performs inference on a slice of data based on a categorical value for a feature.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the data
    feature : str
        Name categorical feature of interest
    category : str
        Name level of the feature of interest    
    trained_model: TrainedModel
        Trained Model object containing a trained model for inference

    Returns
    -------
    results_df : pd.DataFrame
        Dataframe containing the result metrics
    """
    reduced_data = data[data[feature] == category]

    X, y, encoder, lb = process_data(
        reduced_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=trained_model.encoder,
        lb=trained_model.lb
    )

    preds = inference(trained_model.model, X)

    precision, recall, fbeta = compute_model_metrics(y, preds)

    result_dict = {
        'feature': feature,
        'category': category,
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta
    }
    result_df = pd.DataFrame([result_dict])

    return result_df


def save_pretty_df(df, output_path):
    """ Saved a pretty dataframe to a specified file path.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the data
    output_path: str
        String of the output path for the text file
    """
    df.to_markdown(output_path)


def main():
    """
    Main method for executing the slicing script.
    """
    # TODO: bring cat features and model path out to config
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

    CENSUS_DATA_PATH = 'data/census.csv'
    MODEL_OUTPUT_PATH = 'model/trained_model.pkl'
    SLICE_OUTPUT_PATH = './slice_output.txt'

    trained_model = read_model(MODEL_OUTPUT_PATH)

    data = pd.read_csv(CENSUS_DATA_PATH)
    data = data.dropna()

    results_list = []

    for feature in cat_features:
        for category in data[feature].unique():
            result_df = predict_slice(
                data, feature, category, trained_model, cat_features)
            results_list.append(result_df)
    results_df = pd.concat(results_list)
    save_pretty_df(results_df, SLICE_OUTPUT_PATH)


if __name__ == "__main__":
    main()
