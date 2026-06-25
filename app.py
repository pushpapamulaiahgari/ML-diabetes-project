import streamlit as st
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_trained.pkl")
model=joblib.load(MODEL_PATH)


st.title("Diabetes Prediction 😊👍")

st.write("My First ML Project Using Streamlit")

Pregnancies=st.number_input("Enter Pregnancies",min_value=0)
Glucose=st.number_input("Enter Glucose",min_value=0)
BloodPressure=st.number_input("Enter BloodPressure",min_value=0)
SkinThickness=st.number_input("Enter SkinThickness",min_value=0)
Insulin=st.number_input("Enter Insulin",min_value=0)
BMI=st.number_input("Enter BMI",min_value=0.0)
DiabetesPedigreeFunction=st.number_input("Enter DiabetesPedigreeFunction",min_value=0.0)
Age=st.number_input("Enter Age",min_value=0.0)

if st.button("Prediction"):
    input_list=[[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,
    DiabetesPedigreeFunction,Age]]

    final_prediction=model.predict(input_list)     

    if final_prediction[0]==1:
        st.error("Person having Diabetes")
    else:
        st.error("Person Don't have Diabetes")
        