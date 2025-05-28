FROM python:slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN pip install pandas scikit-learn mlflow

RUN git clone https://dagshub.com/nitinjmv/mlops-pipeline.git . \
    && dvc pull models/model.pkl.dvc \
    && ls -lah

COPY src/app .
COPY models/model.pkl .

CMD ["python", "app/app.py"]
