from pathlib import Path

import numpy as np
from pottery.training import logging


def _helper_create_file_paths(path, name):
    if isinstance(path, str):
        path = Path(path)
    samples_path = str(path / f"{name}_samples.npy")
    labels_path = str(path / f"{name}_labels.npy")
    return dict(samples=samples_path, labels=labels_path)


def save_samples(samples, labels, path, name):
    sample_paths = _helper_create_file_paths(path, name)
    np.save(sample_paths["samples"], samples)
    logging.info(f"Data: Storing samples with shape {samples.shape} to file {sample_paths['samples']}")
    np.save(sample_paths["labels"], labels)
    logging.info(f"Data: Storing labels with shape {labels.shape} to file {sample_paths['labels']}")


def load_samples(path, name):
    sample_paths = _helper_create_file_paths(path, name)
    samples = np.load(sample_paths["samples"])
    logging.info(f"Data: Loading samples with shape {samples.shape} from file {sample_paths['samples']}")
    labels = np.load(sample_paths["labels"])
    logging.info(f"Data: Loading samples with shape {labels.shape} from file {sample_paths['labels']}")
    return dict(
        samples=samples,
        labels=labels,
    )