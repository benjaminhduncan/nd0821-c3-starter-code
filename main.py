# Put the code for your API here.
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field

from pathlib import Path
from starter.ml.model import read_model, inference
from starter.ml.data import process_data
from starter.train_model import load_data
import pandas as pd

# Declare the data object with its components and their type.


class CensusData(BaseModel):
    age: int = Field(example=28)
    workclass: str = Field(example='Never-worked')
    fnlwgt: int = Field(example=77516)
    education: str = Field(example='Bachelors')
    education_num: int = Field(alias='education-num', example=13)
    marital_status: str = Field(alias='marital-status', example='Divorced')
    occupation: str = Field(example='Tech-support')
    relationship: str = Field(example='Wife')
    race: str = Field(example='White')
    sex: str = Field(example='Female')
    capital_gain: int = Field(alias='capital-gain', example=2174)
    capital_loss: int = Field(alias='capital-loss', example=0)
    hours_per_week: int = Field(alias='hours-per-week', example=40)
    native_country: str = Field(
        alias='native-country', example='United-States')


app = FastAPI()

# Defines root greeting
@app.get("/")
async def root():
    return {"message": "Welcome to the Salary Classifier API!"}

# This allows inference via POST to the API.
@app.post("/infer")
async def infer(data_model: CensusData):
    data_dict = data_model.dict()
    data = pd.DataFrame(data_dict)
    print(data)

    # TODO: bring model path out to a config
    trained_model = read_model('model/trained_model.joblib')

    # TODO bring categoricals out to a config
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

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=trained_model.encoder,
        lb=trained_model.lb
    )

    raw_preds = inference(trained_model.model, X)
    converted_preds = lb.inverse_transform(raw_preds)[0]

    return converted_preds
