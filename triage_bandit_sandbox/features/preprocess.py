from joblib import dump, load
from pottery.training import logging
from sklearn.preprocessing import StandardScaler

from ..data.io import load_samples


def prepare_preprocessor(data, preprocessor_file="models/preprocessor.joblib"):
    """Prepare the preprocesor and store it to disc for usage in predict."""
    preprocessor = StandardScaler()
    preprocessor.fit(data)
    logging.info(f"Preprocess: Prepared {preprocessor}")
    logging.debug(f"Preprocess: - mean_ {preprocessor.mean_}")
    logging.debug(f"Preprocess: - var_: {preprocessor.var_}")
    dump(preprocessor, preprocessor_file)


def preprocess(data, preprocessor_file="models/preprocessor.joblib"):
    """Importable preprocess function which takes raw samples and applied the preprocessor to it."""
    preprocessor = load(preprocessor_file)
    logging.info(f"Preprocess: Applying {preprocessor} to data with shape {data.shape}")
    preprocessed = preprocessor.transform(data)
    logging.debug(f"Preprocess: Data mean before: {data.mean(axis=1)}")
    logging.debug(f"Preprocess: Data mean after: {preprocessed.mean(axis=1)}")
    logging.debug(f"Preprocess: Data var before: {data.var(axis=1)}")
    logging.debug(f"Preprocess: Data var after: {preprocessed.var(axis=1)}")
    return preprocessed


if __name__ == "__main__":
    data = load_samples("data/features", "train")
    prepare_preprocessor(data["samples"])
