# Put the code for your API here.
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from ml.model import read_model, inference
from ml.data import process_data

# Declare the data object with its components and their type.

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class CensusData(BaseModel):
    """
    CensusData model defining the expected input fields for inference.
    """
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
    """
    Function defining the root api get return message.

    Returns
    -------
    message : dict
        Welcome message dictionary.
    """
    return {"msg": "Welcome to the Salary Classifier API!"}

# This allows inference via POST to the API.


@app.post("/infer")
async def infer(data_model: CensusData):
    """
    Function defining the inference post endpoint

    Inputs
    ------
    data_model: CensusData
        Data model input for inference
    Returns
    -------
    predictions : str
        String of predicted income classification.
    """
    data_dict = data_model.dict()
    data = pd.DataFrame([data_dict])

    # TODO: bring model path out to a config
    trained_model = read_model('model/trained_model.pkl')

    # TODO bring categoricals out to a config
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=trained_model.encoder,
        lb=trained_model.lb
    )

    raw_preds = inference(trained_model.model, X)
    converted_preds = lb.inverse_transform(raw_preds)[0]

    return {"output": converted_preds}
