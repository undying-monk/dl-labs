import mlflow
from sklearn.metrics import accuracy_score

def wrapper_logging(training_loop_func):
    def inner(*args, **kwargs):
        with mlflow.start_run():
            params = args[-1]
            mlflow.log_params(params)

            model, y_pred, y_test = training_loop_func(*args, **kwargs)

            info = mlflow.sklearn.log_model(sk_model=model, name="iris_model")
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy",accuracy)

            # Optional: Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")
            return model, y_pred, y_test
    return inner