import mlflow
from sklearn.metrics import accuracy_score
from mlflow import MlflowClient

model_name = "iris_model"

def wrapper_logging(training_loop_func, enable_system_metrics):
    def inner(*args, **kwargs):
        with mlflow.start_run(log_system_metrics=enable_system_metrics):
            params = args[-1]
            mlflow.log_params(params)

            model, y_pred, y_test = training_loop_func(*args, **kwargs)

            info = mlflow.sklearn.log_model(sk_model=model, registered_model_name=model_name)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy",accuracy)

            new_version = info.registered_model_version

            # Optional: Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")

            alias_tag = "production"
            set_alias_tag(new_version, alias_tag)

            return model, y_pred, y_test
    return inner

def set_alias_tag(version, alias):
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name,alias=alias,version=str(version))
    print(f"Successfully registered and set version {version} to '@{alias}'")

def get_model_uri(alias):
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias)
    return f"models:/{model_version.name}/{model_version.version}"

def download_artifacts(alias):
    client = MlflowClient()
    print("model_version", model_name, alias)
    model_version = client.get_model_version_by_alias(model_name, alias)
    underlying_artifact_uri = model_version.source
    print(f"Underlying Storage Path: {underlying_artifact_uri}")

    download_uri = client.get_model_version_download_uri(
        name=model_name, version=model_version.version
    )
    print(f"Registry Download URI: {download_uri}")
    return mlflow.artifacts.download_artifacts(artifact_uri=download_uri)

def load_tags_by_alias(alias):
    client = MlflowClient()
    model_info = client.get_model_version_by_alias(model_name, alias)
    model_tags = model_info.tags
    print("Model Tags",model_tags)
    return model_tags

def load_model_by_alias(alias):
    # Get the model version using a model URI
    model_uri = f"models:/{model_name}@{alias}"
    print("model_uri", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model

def load_model_by_version(version):
# Get the model version using a model URI
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model