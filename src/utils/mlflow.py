from src.utils.commons import load_params, logging_setup
import mlflow
import dagshub

params = load_params(params_path='./params.yaml')
experiment_name = params['mlflow']['experiment_name']
repo_owner = params['mlflow']['repo_owner']
repo_name = params['mlflow']['repo_name']
tracking_uri = params['mlflow']['tracking_uri']

logger = logging_setup('mlflow')
logger.debug('experiment_name %s', experiment_name)
logger.debug('repo_owner %s', repo_owner)
logger.debug('repo_name %s', repo_name)
logger.debug('tracking_uri %s', tracking_uri)

def dagshub_integration():
    mlflow.set_tracking_username(os.getenv("DAGSHUB_USERNAME"))
    mlflow.set_tracking_password(os.getenv("DAGSHUB_TOKEN"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()