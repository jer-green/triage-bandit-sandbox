import numpy as np
from joblib import load
from pottery import model, pottery
from pottery.training import logging

from .features.preprocess import preprocess


@pottery
@model("models/mlp.joblib")
def predict(data, model_file):
    logging.info(f"Predict: Data shape: {data.shape}, Model file: {model_file}")
    model = load(model_file)
    logging.info(f"Predict: Loaded model {model}")
    preprocessed = preprocess(data)
    prediction = model.predict(preprocessed)
    logging.info(f"Predict: Predicted labels with shape: {prediction.shape}")
    return prediction


@predict.preprocessor
def preprocess_request(request):
    # TODO: code smell:
    if isinstance(request, str):
        return  np.atleast_2d(np.fromstring(request.strip("[]"), dtype=float, sep=","))
    else:
        return request

@predict.postprocessor
def create_request(data):
    # TODO: code smell:
    # What to do here to distinguish between running on the service and running as imported function?
    return {k: str(v) for k, v in enumerate(data)}