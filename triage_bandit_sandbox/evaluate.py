import numpy as np
import pandas as pd
from pottery.training import logging, track_experiment
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)

from .data.io import load_samples
from .predict import predict


def evaluate_metrics(y_true, y_pred):
    precision, recall, fscore = precision_recall_fscore_support(y_true, y_pred, average="macro")[:3]
    return dict(
        precision=precision,
        recall=recall,
        fscore=fscore,
    )

def to_json(metrics, filename):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_json(filename)


def evaluate(
    samples, labels
):
    """Evaluate model given a set of samples and ground truth labels"""
    logging.info(f"Evaluate: Evaluating training samples {samples.shape} against labels {labels.shape}")
    y_pred = predict(samples)

    # TODO: Code smell...
    y_pred = np.array([
        np.fromstring(v.strip("[]"), dtype=float, sep=",")
        for v in y_pred.values()
    ])

    metrics = evaluate_metrics(labels, y_pred)
    logging.info(f"Evaluate: metrics: {metrics}")
    
    testing_metrics_file = "test.json"
    logging.info(f"Evaluate: storing metrics to file: {testing_metrics_file}")
    to_json(metrics, testing_metrics_file)
    
    testing_report = classification_report(labels, y_pred)
    logging.debug(f"Evaluate: classifciation report:\n{testing_report}")

    track_experiment.log_metric("precision", metrics["precision"])
    track_experiment.log_metric("recall", metrics["recall"])
    track_experiment.log_metric("fscore", metrics["fscore"])

    return metrics



if __name__ == "__main__":
    testing_data = load_samples("data/features", "test")
    evaluate(testing_data["samples"], testing_data["labels"])
    pass