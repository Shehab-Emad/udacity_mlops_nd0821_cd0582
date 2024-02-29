from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome_msg": "welcome to the Model"}
    

# A function to test the post on a predicted value of Salaray < 50K
def test_post_under_50k():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json() == {"salary": "< 50k"}

# A function to test the post on a predicted value of Salaray > 50K
def test_post_above_50k():
    data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json() == {"salary": "> 50k"}

# # A function to test the malformed payload
# def test_malformed_payload():
#     data = {}
#     r = client.post("/predict", json=data)
#     assert r.status_code == 422