import mlflow

mlflow.set_experiment("MLflow Quickstart")

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import wrapper_logging, load_model_by_alias,load_model_by_version, get_preprocessing

def get_columns():
    return datasets.load_iris().feature_names

def load_dataset():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train,y_train, X_test,y_test

def get_hyperparameters():
    return {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }

def train_model(X_train,y_train, params):
    # Enable autologging for scikit-learn
    # mlflow.sklearn.autolog()

    # Just train the model normally
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    return lr, params

def predict(model, X, y):
    y_pred = model.predict(X)
    return model, (y, y_pred)

def load_model(file_path):
    loaded_model = mlflow.pyfunc.load_model(file_path)
    return loaded_model

def visual_df(X_test, y_test, model, columns):
    y_pred = model.predict(X_test)
    df = pd.DataFrame(X_test, columns=columns)
    df["pred"] = y_pred
    df["actual"] = y_test
    print(df[:4])

# can use decorator in here
def training_loop(X_train,y_train,X_test,y_test, params):
    model, params = train_model(X_train,y_train, params)
    y_pred = model.predict(X_test)
    return model, y_pred, y_test

def load_model(X_test,y_test):
    model = load_model_by_alias("production")
    idx = 0

    y_pred = model.predict(X_test[idx].reshape(1, -1))
    print("Predict", X_test[idx], y_pred[-1], y_test[idx])

def main():
    X_train,y_train, X_test,y_test = load_dataset()
    params = get_hyperparameters()

    # wrapped_training_loop = wrapper_logging(training_loop, enable_system_metrics=True)
    # model, y_pred, y_test = wrapped_training_loop(X_train,y_train,X_test,y_test, params)

    columns = get_columns()
    # visual_df(X_test, y_test, model, columns)

    load_model(X_test,y_test)

    df = pd.DataFrame(X_train, columns=columns)
    numeric_columns = df.select_dtypes(include=['float64','int64']).columns
    categorical_columns = df.select_dtypes(include=['object','category']).columns
    ordinal_columns = []
    preprocessing = get_preprocessing(numeric_columns, categorical_columns, ordinal_columns)
    preprocessing.fit_transform(X_train)

if __name__ == "__main__":
    main()