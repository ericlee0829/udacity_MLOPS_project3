from fastapi.testclient import TestClient
from main import app
from json import dumps

client = TestClient(app)


def test_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "welcome": "This is test api for MLDevOps Nanodegree for Deploying a ML model on Heroku with FastAPI.\n \t See the census.csv for inference"
    }


def test_inference_for_less_and_equal_than_50k():
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 252803,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    r = client.post("/", json=data)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_post_sample_greater_than_50K():
    data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 132996,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5178,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States",
    }
    r = client.post("/", json=data)
    assert r.status_code == 200
    assert r.json() == ">50K"
