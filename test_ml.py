import pytest
# TODO: add necessary import
from ml.model import train_model
from ml.model import inference
from ml.data import process_data

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# TODO: implement the first test. Change the function name and input as needed
def test_train_model_type():
    """
    test train_model() returns the correct model type
    """
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_length():
    """
    # test inference() function returns expected length
    """
    class DummyModel:
        def predict(self, X):
            return [0] * len(X)

    X = [[1, 2], [3, 4]]

    preds = inference(DummyModel(), X)

    assert len(preds) == len(X)


# TODO: implement the third test. Change the function name and input as needed
def test_process_data():
    """
    # verifies process_data() function runs and returns the correct number of rows
    """
    # Your code here
    df = pd.DataFrame({
        "age": [30],
        "workclass": ["Private"],
        "education": ["Bachelors"],
        "marital-status": ["Single"],
        "occupation": ["Tech"],
        "relationship": ["Not-in-family"],
        "race": ["White"],
        "sex": ["Male"],
        "native-country": ["United-States"],
        "salary": ["<=50K"]
    })

    cat_features = [
        "workclass","education","marital-status",
        "occupation","relationship","race","sex","native-country"
    ]

    X, y, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == 1
    assert y.shape[0] == 1
