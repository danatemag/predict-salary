from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_main():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_0():
    response = client.post("/", json={
        "age": 24,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

    assert response.status_code == 200
    assert response.json() == {"prediction": 0}


def test_predict_1():
    response = client.post("/", json={
        "age": 58,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 93664,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husban",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    })

    assert response.status_code == 200
    assert response.json() == {"prediction": 1}
