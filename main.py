import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data

app = FastAPI()

model = pickle.load(open('model/model.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))
lb = pickle.load(open('model/lb.pkl', 'rb'))

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

class DataRow(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

@app.get("/")
def read_root() -> dict:
    return {"Hello": "World"}


@app.post("/")
def inference(row: DataRow) -> dict:
    global model, encoder, lb, cat_features
    
    obj = row.dict()
    newobj = { k.replace('_', '-'): v for k, v in obj.items() }
    df = pd.DataFrame([newobj])

    print(df.values[0])
    X, _, _, _ = process_data(df, cat_features, training=False, encoder=encoder, lb=lb)
    print(X)
    preds = model.predict(X)
    print(preds)

    return {
        "prediction": preds[0].item()
    }
