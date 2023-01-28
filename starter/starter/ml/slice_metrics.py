import pandas as pd
import sys
import os
import joblib
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


def slice_metrics(cat_features):
    file_dir = os.path.dirname(os.path.abspath("__file__"))
    sys.path.insert(0, file_dir)

    data = pd.read_csv(file_dir + "/../data/clean_census.csv")
    model = joblib.load(file_dir + "/../model/census_rfmodel.pkl")
    encoder = joblib.load(file_dir + "/../model/census_encoder.pkl")
    lb = joblib.load(file_dir + "/../model/census_lb.pkl")

    os.makedirs(file_dir + "/../slice_metrics", exist_ok=True)
    flie = open(file_dir + "/../slice_metrics/slice_output.txt", "w")
    for slice_feature in cat_features:
        for elem in data[slice_feature].unique():
            data_temp = data[data[slice_feature] == elem]
            X, y, _, _ = process_data(
                data_temp,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            predicts = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, predicts)
            txt_line = f"{slice_feature} - {elem}: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}\n"
            flie.write(txt_line)
    flie.close()
