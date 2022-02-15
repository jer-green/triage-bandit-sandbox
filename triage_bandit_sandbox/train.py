import numpy as np
import pandas as pd
from joblib import dump
from pottery.training import logging, track_experiment
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from .data.io import load_samples
from .evaluate import evaluate_metrics, to_json
from .features.preprocess import preprocess


def train(alpha, max_iter, samples, labels, model_file):
    """
    Train a simple MLPClassifier using sklearn.

    Replace code in here with your chosen ML model to be trained on the input data.
    """
    track_experiment.log_param("alpha", alpha)
    track_experiment.log_param("max_iter", max_iter)

    clf = MLPClassifier(alpha=alpha, max_iter=max_iter)

    logging.info(f"Train: Model to train {clf}")
    logging.info(f"Train: Training data shape {samples.shape}")
    logging.info(f"Train: Training labels shape {labels.shape}")

    X = preprocess(samples)

    np.random.seed(42)
    clf.fit(X, labels)

    logging.info(f"Train: Storing model to {model_file}")
    dump(clf, model_file)

    predicted_qualities = clf.predict(X)

    metrics = evaluate_metrics(labels, predicted_qualities)
    logging.info(f"Train: metrics: {metrics}")

    training_metrics_file = "train.json"
    logging.info(f"Train: storing metrics to file: {training_metrics_file}")
    to_json(metrics, training_metrics_file)

    training_report = classification_report(labels, predicted_qualities)
    logging.debug(f"Train: classifciation report:\n{training_report}")

    track_experiment.log_metric("train_precision", metrics["precision"])
    track_experiment.log_metric("train_recall", metrics["recall"])
    track_experiment.log_metric("train_fscore", metrics["fscore"])


if __name__ == "__main__":
    data = load_samples("data/features", "train")
    train(1, 1000, data["samples"], data["labels"], "models/mlp.joblib")
