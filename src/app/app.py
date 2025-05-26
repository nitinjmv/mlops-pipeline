from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

DATA_PATH = "data/processed/processed_v1.csv"
MODEL_PATH = "models/model_v1.pkl"

@app.route('/')
def index():
    df = pd.read_csv(DATA_PATH)
    return render_template("index.html", tables=[df.head().to_html(classes='data')], titles=df.columns.values)

@app.route('/train', methods=['POST'])
def train():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return render_template("index.html", message=f"✅ Model trained with accuracy: {acc:.2f}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

