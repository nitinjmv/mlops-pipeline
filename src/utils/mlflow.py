from commons import load_params
import mlflow
import dagshub

params = load_params(params_path='./params.yaml')
experiment_name = params['mlflow']['experiment_name']
repo_owner = params['mlflow']['repo_owner']
repo_name = params['mlflow']['repo_name']
tracking_uri = params['mlflow']['tracking_uri']

def dagshub_integration():
    dagshub.init(repo_owner = repo_owner, repo_name = repo_name, mlflow=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.autolog()
    mlflow.set_experiment(experiment_name)