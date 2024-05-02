import requests

BASE_URI = 'https://predict-salary-w3jd.onrender.com'

response = requests.post(BASE_URI + "/", json={
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
