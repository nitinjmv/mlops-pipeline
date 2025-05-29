import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_metric, log_param, log_artifact, set_experiment, start_run, mlflow
from src.utils.commons import load_params, logging_setup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging_setup('model_building')
mlflow.autolog()

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        mlflow.log_param("n_estimators", params['n_estimators'])
        mlflow.log_param("random_state", params['random_state'])
        mlflow.log_metric("accuracy", clf.score(X_train, y_train))
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise



def main():
    try:
        mlflow.set_experiment("model_building_experiment")
        with mlflow.start_run() as run:
            params = load_params('./params.yaml')['model_building']
            set_experiment(f'model-building: {params["experiment_name"]}')
            log_param("n_estimators", params["n_estimators"])
            log_param("random_state", params["random_state"])

            # ✅ Load cleaned raw text data (not TF-IDF)
            train_data = load_data('./data/interim/train_processed.csv')
            X_train = train_data['text'].values
            y_train = train_data['target'].values

            # ✅ Build full pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    random_state=params["random_state"]
                ))
            ])

            logger.debug('Fitting pipeline...')
            pipeline.fit(X_train, y_train)

            log_metric("num_training_samples", len(X_train))
            log_metric("train_accuracy", pipeline.score(X_train, y_train))

            model_save_path = 'models/model.pkl'
            save_model(pipeline, model_save_path)
            log_artifact(model_save_path)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name="SpamModel"
            )

            if os.path.exists("reports/metrics.json"):
                log_artifact("reports/metrics.json")
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
