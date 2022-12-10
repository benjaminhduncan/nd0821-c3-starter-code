from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Initialize model
    gbc = GradientBoostingClassifier()

    # Select model parameters for tuning
    parameters = {
        "n_estimators": [5,50,250,500],
        "max_depth": [1,3,5,7,9],
        "learning_rate": [0.01,0.1,1,10,100]
    }

    # Perform a hyperparameter tuning grid search
    cv = GridSearchCV(gbc, parameters, cv=5)
    
    cv.fit(X_train, y_train)

    # gbc.fit(X_train, y_train)

    print(type(cv.best_estimator_))
    print(cv.best_params_)
    
    return cv.best_estimator_
    
    # return gbc

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    preds = model.predict(X)

    return preds
