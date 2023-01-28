# Script to train machine learning model.

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from joblib import dump

# Add the necessary imports for the starter code.
import os
import sys
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

file_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, file_dir)


# Add code to load in the data.
logger.info("Read clean_census.csv")
data = pd.read_csv(file_dir + "/../data/clean_census.csv")
# data = pd.read_csv(file_dir + '/../data/census.csv', sep=', ', engine='python')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Spliting data")
train, test = train_test_split(data, test_size=0.20)

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

logger.info("Processing train data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logger.info("Processing test data")
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model
logger.info("Training RandomForeset model")
rf_model = train_model(X_train, y_train)

y_pred = inference(rf_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logger.info(f"Precision: {precision: .1f}, Recall: {recall: .1f}, Fbeta: {fbeta: .1f}")

logger.info("Saving model")
dump(rf_model, file_dir + "/../model/census_rfmodel.pkl")
dump(encoder, file_dir + "/../model/census_encoder.pkl")
dump(lb, file_dir + "/../model/census_lb.pkl")
