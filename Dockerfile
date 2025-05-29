FROM python:slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && pip install --upgrade pip

RUN pip install pandas scikit-learn dvc mlflow nltk joblib flask

RUN python -m nltk.downloader punkt stopwords

RUN git clone https://dagshub.com/nitinjmv/mlops-pipeline.git . \
    && dvc pull models/model.pkl.dvc \
    && ls -lah

COPY src/app .
COPY models/ ./models/

CMD ["python", "app.py"]
