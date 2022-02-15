import numpy as np
from pottery.training import logging
from sklearn.model_selection import train_test_split

from ..data.io import load_samples, save_samples
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
# from babylon_pgm.constants import LabelType, Gender, BOOLEAN_STATES
#
# from clinical_validation_sandbox.models import CaseCard
# from triage_models_sandbox import (
#     TriageData,
#     TriageDataset,
#     TriageClassifier,
#     CrueltyDiseasesCoarseGrainer,
#     MultiLabelEncoder,
#     TriageBandit,
# )
# from inference.engine import InferenceEngine, FeatureFlags
# from utils.inference_model_from_json import JSONDatasetToInferenceModelConverter


def create_train_test_split():
    pass


# def xx_create_train_test_split(
#         train_set_size,
#         random_state,
#         in_path,
#         in_name,
#         out_path,
# ):
#     """splits the data into training and testing sets"""
#     data = load_samples(in_path, in_name)
#     X = data["samples"]
#     y = data["labels"]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, train_size=train_set_size, random_state=random_state
#     )
#
#     logging.info(f"Featurize: Splitting dataset with shape {X.shape} into training and testing samples")
#     logging.info(f"Featurize:  - Train set shape {X_train.shape}")
#     logging.info(f"Featurize:  - Test set shape {X_test.shape}")
#
#     save_samples(X_train, y_train, out_path, "train")
#     save_samples(X_test, y_test, out_path, "test")


if __name__ == "__main__":
    # create_train_test_split(0.8, 42, "data/raw", "example", "data/features")
    create_train_test_split()
