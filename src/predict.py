import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

def predict(sample):
    df = pd.DataFrame([sample])
    return model.predict(df)