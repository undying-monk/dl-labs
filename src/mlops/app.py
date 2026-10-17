from contextlib import asynccontextmanager
from typing import Annotated
import os
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .utils.dataprocessing import prepare_features
from .utils.utils import load_model_by_alias, get_model_uri
from .model import FEATURES

MODEL_URI = os.environ.get(
    "MODEL_URI",
    get_model_uri("production"),
)


Measurement = Annotated[
    float, Field(gt=0, allow_inf_nan=False, strict=True)
]


class IrisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: Measurement = Field(alias="sepal length (cm)")
    sepal_width: Measurement = Field(alias="sepal width (cm)")
    petal_length: Measurement = Field(alias="petal length (cm)")
    petal_width: Measurement = Field(alias="petal width (cm)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_by_alias("production")
    yield
    del app.state.model


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(inputs: IrisInput):
    df = pd.DataFrame([inputs.model_dump(by_alias=True)])
    print(df)
    features = prepare_features(df, FEATURES=FEATURES)
    predictions = app.state.model.predict(features)

    return {
        "prediction": int(predictions[0]),
        "model_uri": MODEL_URI,
    }