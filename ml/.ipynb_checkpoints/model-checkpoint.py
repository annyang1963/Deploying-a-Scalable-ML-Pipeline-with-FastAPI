import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from joblib import dump

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
   # TODO: implement the function
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
    

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
    # TODO: implement the function

    '''
    #ATTEMPT AT APPLYING LABELS?  I need y_slice to preds to match and only one is binarized
    # Run model predictions
    preds = model.predict(X)

    # Apply labels to predictions
    preds_with_labels = [apply_label(pred) for pred in preds]

    return preds_with_labels
    '''


def inference(model, X):
    """ Run model inferences and return the predictions."""
    preds = model.predict(X)
    #add labels
    return preds


def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function
    with open(path, 'rb') as f:
        #model = pickle.load(f)
        return pickle.load(f)
    #return model


#COMMENTED OUT FOR TESTING#########################################################################################
'''
def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    X_slice, y_slice, _, _ = process_data(
        data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        #lb=lb added in to test
        #lb=lb
    )
 
    #preds = model.predict(X_slice)
    preds = inference(model, X_slice)
    
    # Calculate precision, recall, and F1 score for the slice
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
    '''
#END OF GOOD CODE########################################################################################

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # Process data for slices, including label binarization
    X_slice, y_slice, _, _ = process_data(
        data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb  # Binarize the label for slices
    )

    # Predict using the model
    preds = inference(model, X_slice)
    
    # Calculate precision, recall, and F1 score for the slice
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta