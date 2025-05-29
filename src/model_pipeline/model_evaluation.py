import os
import numpy as np
import pandas as pd
import pickle
import json
import hashlib
from datetime import datetime
import platform
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from mlflow import start_run, set_experiment, log_param, log_metric, log_artifact, mlflow

from src.utils.commons import load_params, logging_setup

logger = logging_setup('model_evaluation')
mlflow.autolog()


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def hash_file(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.debug('Model evaluation metrics calculated')
        return metrics_dict, y_pred
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    try:
        set_experiment("model_evaluation_experiment")
        with start_run() as run:
            params = load_params(params_path='./params.yaml')['model_building']
            clf = load_model('./models/model.pkl')
            test_data_path = './data/interim/test_processed.csv'
            test_data = load_data(test_data_path)

            X_test = test_data['text'].astype(str).values
            y_test = test_data['target'].values

            metrics, y_pred = evaluate_model(clf, X_test, y_test)

            # Log identifiers
            model_uri = f"runs:/{run.info.run_id}/model"
            log_param("evaluated_model_uri", model_uri)
            log_param("eval_data_hash", hash_file(test_data_path))
            log_param("evaluation_timestamp", datetime.now().isoformat())
            log_param("python_version", platform.python_version())

            # Log model parameters again for traceability
            log_param("n_estimators", params['n_estimators'])
            log_param("random_state", params['random_state'])

            # Log metrics
            for key, value in metrics.items():
                log_metric(key, value)

            # Save metrics
            os.makedirs("reports", exist_ok=True)
            metrics_path = 'reports/metrics.json'
            save_metrics(metrics, metrics_path)
            log_artifact(metrics_path)

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            cm_path = "reports/confusion_matrix.png"
            plt.savefig(cm_path)
            log_artifact(cm_path)
            plt.close()

            # Log classification report
            report_text = classification_report(y_test, y_pred)
            report_path = "reports/classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report_text)
            log_artifact(report_path)

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
