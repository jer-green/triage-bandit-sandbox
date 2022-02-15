import numpy as np
from pottery.training import logging
from sklearn.datasets import make_classification

from .io import save_samples


def download_case_cards():
    # just manually downloaded case cards for the moment
    pass


def download_data(count, n_dims, n_classes, outpath, name):
    """Download data set to the data folder for further usage"""
    dataset, ground_truth = make_classification(
        n_samples=count,
        n_features=n_dims,
        n_informative=2,
        n_redundant=2,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=42,
    )
    logging.info("Download_data: Creating example classification dataset")
    logging.info(f"Download_data:  - samples: {dataset.shape}")
    logging.info(f"Download_data:  - labels: {ground_truth.shape}")

    save_samples(dataset, ground_truth, outpath, name)


if __name__ == "__main__":
    # download_data(1000, 25, 2, "data/raw/", "example")
    download_case_cards()
