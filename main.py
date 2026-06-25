from fastapi import FastAPI
from pydantic import BaseModel

import joblib
app=FastAPI()
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_project", "model_trained.pkl")

print(MODEL_PATH)

model = joblib.load(MODEL_PATH)
class DiabetesInput(BaseModel):
    Pregnancies:int
    Glucose :int
    BloodPressure:int
    SkinThickness:int
    Insulin:float
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int

@app.get('/')
def home():
    return{"message":"My first ML Project Using FastAPI"}

@app.post("/predict")
def prediction (data:DiabetesInput):
    input_list=[[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]
    final_prediction=model.predict(input_list)

    result = 'Diabetes' if final_prediction[0] == 1 else 'No Diabetes'
    return{
       'prediction':result
     }
