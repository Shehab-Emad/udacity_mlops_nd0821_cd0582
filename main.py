# Put the code for your API here.

import os
import numpy as np
import pandas as pd
#Import libraries related to fastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
#Import the inference function to be used to predict the values
from starter.ml.model import inference
from starter.ml.data import process_data

#Import the model to be used to predict
model = pd.read_pickle(r"model/model.pkl")
Encoder = pd.read_pickle(r"model/encoder.pkl")
# ml_models = {}

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load the ML model
#     ml_models["model"] = pd.read_pickle(r"model/model.pkl")
#     ml_models["Encoder"] = pd.read_pickle(r"model/encoder.pkl")
#     yield
#     # Clean up the ML models and release the resources
#     ml_models.clear()

#Initial a FastAPI instance
app = FastAPI()
# app = FastAPI(lifespan=lifespan)



# pydantic models
class DataIn(BaseModel):
    #The input should be alist of 108 values 
    age : int = 39
    workclass : str =  "State-gov"
    fnlgt : int = 77516
    education : str = "Bachelors"
    education_num : int = 13
    marital_status : str = "Never-married"
    occupation : str = "Adm-clerical"
    relationship : str = "Not-in-family"
    race : str = "White"
    sex : str = "Male"
    capital_gain : int = 2174
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = "United-States"

class DataOut(BaseModel):
    #The forecast output will be either >50K or <50K 
    salary: str = "> 50k"

#Adding a Welcome message to the initial page
@app.get("/")
async def root():
    return {"welcome_msg": "welcome to the Model"}


@app.post("/predict", response_model=DataOut, status_code=200)
def get_prediction(payload: DataIn):
    #Reading the input data
    age = payload.age
    workclass = payload.workclass
    fnlgt = payload.fnlgt
    education = payload.education
    education_num = payload.education_num
    marital_status = payload.marital_status
    occupation = payload.occupation
    relationship = payload.relationship
    race = payload.race
    sex = payload.sex
    capital_gain = payload.capital_gain
    capital_loss = payload.capital_loss
    hours_per_week = payload.hours_per_week
    native_country = payload.native_country
    #Converted the inputs into Dataframe to be processed 
    df = pd.DataFrame([{"age" : age,
                        "workclass" : workclass,
                        "fnlgt" : fnlgt,
                        "education" : education,
                        "education-num" : education_num,
                        "marital-status" : marital_status,
                        "occupation" : occupation,
                        "relationship" : relationship,
                        "race" : race,
                        "sex" : sex,
                        "capital-gain" : capital_gain,
                        "capital-loss" : capital_loss,
                        "hours-per-week" : hours_per_week,
                        "native-country" : native_country}])
    # Process the data with the process_data function.
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
    X_processed, y_processed, encoder, lb = process_data(df, categorical_features=cat_features, training=False,encoder=Encoder)
    # X_processed, y_processed, encoder, lb = process_data(df, categorical_features=cat_features, training=False, encoder=ml_models["Encoder"])
    # Inference function for prediction
    pred = inference(model, X_processed)
    # pred = inference(ml_models["model"], X_processed)
    
    out = '< 50k' if pred == 0 else '> 50k'
    return {"salary": out}