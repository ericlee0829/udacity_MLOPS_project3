import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pytest
import os
import sys

sys.path.insert(0, "starter/starter")
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

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


@pytest.fixture
def data():
    """
    Get Census data
    """
    df = pd.read_csv("starter/data/clean_census.csv")
    return df


def test_process_data(data):
    train, test = train_test_split(data, test_size=0.20)

    X, y, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert len(X) == len(y)


def test_train_model(data):
    train, test = train_test_split(data, test_size=0.20)

    X, y, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    # Check if this is a classification model
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(data):
    y = [1, 1, 0, 0]
    predicts = [0, 1, 1, 0]
    precision, recall, fbeta = compute_model_metrics(y, predicts)
    print(precision, recall, fbeta)
    assert (
        abs(precision - 0.5) < 0.01
        and abs(recall - 0.5) < 0.01
        and abs(fbeta - 0.5) < 0.01
    )


def test_inference(data):
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    pred = inference(model, X)
    # Check if pred.shape is similar to y.shape
    assert y.shape == pred.shape
