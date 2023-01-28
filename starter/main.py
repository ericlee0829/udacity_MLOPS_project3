# Put the code for your API here.
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel, Field
import pandas as pd
from joblib import load
import sys
import os

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

data = pd.read_csv("starter/data/clean_census.csv")
model = load("starter/model/census_rfmodel.pkl")
encoder = load("starter//model/census_encoder.pkl")
lb = load("starter/model/census_lb.pkl")

# Instantiate the app.
app = FastAPI()


class DataSample(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example="State-gov")
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example="Bachelors")
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example="Never-married")
    occupation: str = Field(None, example="Adm-clerical")
    relationship: str = Field(None, example="Not-in-family")
    race: str = Field(None, example="White")
    sex: str = Field(None, example="Female")
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example="United-States")


# Define a GET on the specified endpoint.
@app.get("/")
async def welcome_message():
    return {
        "welcome": "This is test api for MLDevOps Nanodegree for Deploying a ML model on Heroku with FastAPI.\n \t See the census.csv for inference"
    }


@app.post("/")
async def inference_from_sample(sample: DataSample):
    sample = {key.replace("_", "-"): [value] for key, value in sample.__dict__.items()}
    df = pd.DataFrame.from_dict(sample)
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    predict = inference(model, X)[0]
    return "<=50K" if predict == 0 else ">50K"


# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull -f") != 0:
#         exit("dvc pull failed")vi
#     os.system("rm -r .dvc .apt/usr/lib/dvc")
