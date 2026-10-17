import shutil
import mlflow.artifacts
import mlflow
from utils.utils import download_artifacts

mlflow.set_experiment("MLflow Quickstart")
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# downloaded = mlflow.artifacts.download_artifacts(
#     artifact_uri="models:/my-classifier/1",
# )

downloaded = download_artifacts("production")

# Creates model/; this destination should not already exist.
shutil.copytree(downloaded, "./model")