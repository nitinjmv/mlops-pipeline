import streamlit as st
import requests

st.title("Iris Species Predictor")

sepal_length = st.slider("Sepal Length", 4.0, 8.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5)
petal_length = st.slider("Petal Length", 1.0, 7.0)
petal_width = st.slider("Petal Width", 0.1, 2.5)

features = [sepal_length, sepal_width, petal_length, petal_width]

if st.button("Predict"):
    response = requests.post("http://localhost:8000/predict", json={"features": features})
    prediction = response.json()["prediction"]
    st.write(f"Predicted class: {prediction[0]}")