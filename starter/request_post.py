import json
import requests

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

response = requests.post("https://mldevopnd-app.herokuapp.com/", json=data)

print(response.status_code)
print(response.json())
