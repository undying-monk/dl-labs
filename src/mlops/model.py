import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

def get_hyperparameters():
    return {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }

def init_model(params):
    # Enable autologging for scikit-learn
    # mlflow.sklearn.autolog()

    # Just train the model normally
    lr = LogisticRegression(**params)
    return lr

def load_model(file_path):
    loaded_model = mlflow.pyfunc.load_model(file_path)
    return loaded_model

def training_loop(X_train,y_train,X_test,y_test, params):
    model = init_model(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, y_test

def test_one_example(model, X_test, y_test):
    idx = 0
    y_pred = model.predict(X_test[idx].reshape(1, -1))
    print("Predict", X_test[idx], y_pred[-1], y_test[idx])