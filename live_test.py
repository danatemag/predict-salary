import requests
import logging


logging.basicConfig(level=logging.DEBUG)
BASE_URI = 'https://predict-salary-w3jd.onrender.com'
INFERENCE_URI = BASE_URI + "/"


data_input = {
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
}

response = requests.post(INFERENCE_URI, json=data_input)
prediction = response.json()['prediction']

print(f'POST {INFERENCE_URI}')
print(f'Input {data_input}')
print(f'Response status code: {response.status_code}')
print(f'Inference result {prediction}')
