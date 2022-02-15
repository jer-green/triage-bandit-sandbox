from typing import Optional
import numpy as np
from scipy.stats import mode

from babylon_pgm.triage_models.triage_decisions import (
    TRIAGE_MODEL_DECISION_TYPE,
    TriageModelDecisionDefault,
)

from ..interfaces import PreparedDataTransformer, PreparedData


def _check_correct_decisions_type(correct_decisions):
    assert isinstance(correct_decisions, list)
    assert isinstance(correct_decisions[0], list)
    check_flag = False
    for t in TRIAGE_MODEL_DECISION_TYPE.__args__:
        if isinstance(correct_decisions[0][0], t):
            check_flag = True
            break
    if check_flag is False:
        raise AssertionError("Type not recognized.")

class MultiClassEncoder(PreparedDataTransformer):
    """
    Encode correct triage decisions into labels for multi-class classifiers.
    The mode of doctors' outcomes is used as the correct decision.
    """
    def __init__(self):
        pass

    def fit(self, dataset: PreparedData):
        pass

    def transform(
            self, dataset: PreparedData, train: Optional[bool] = True
    ) -> PreparedData:
        """
        Transform a PreparedData object into a PreparedData object, where
        correct_decision are encoded as labels for a multi-class classifier.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object.
        """
        if dataset.correct_decisions is not None:
            _check_correct_decisions_type(dataset.correct_decisions)
            y = np.zeros(len(dataset),)
            for i, decisions in enumerate(dataset.correct_decisions):
                y[i] = mode([d.index for d in decisions])[0][0]
            dataset.correct_decisions = y.astype(int)
        return dataset

    def fit_transform(
            self, dataset: PreparedData, train: Optional[bool] = True
    ) -> PreparedData:
        """
        Transform a PreparedData object into a PreparedData object, where
        correct_decision are encoded as labels for a multi-class classifier.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object.
        """
        return self.transform(dataset)


class MultiLabelEncoder(PreparedDataTransformer):
    """
    Encode correct triage decisions into labels for multi-label classifiers.
    All doctor's outcomes are considered as valid decisions.
    """
    def __init__(
            self,
            triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
    ):
        self._triage_decisions = triage_decisions
        self.num_classes = len(self._triage_decisions)

    def fit(self, dataset: PreparedData):
        pass

    def transform(
            self, dataset: PreparedData, train: Optional[bool] = True
    ) -> PreparedData:
        """
        Transform a PreparedData object into a PreparedData object, where
        correct_decision are encoded as labels for a multi-label classifier.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object.
        """
        if dataset.correct_decisions is not None:
            _check_correct_decisions_type(dataset.correct_decisions)
            y = np.zeros((len(dataset), self.num_classes))
            for i, decisions in enumerate(dataset.correct_decisions):
                y[i, [d.index for d in decisions]] = 1
            dataset.correct_decisions = y
        return dataset

    def fit_transform(
            self, dataset: PreparedData, train: Optional[bool] = True
    ) -> PreparedData:
        """
        Transform a PreparedData object into a PreparedData object, where
        correct_decision are encoded as labels for a multi-label classifier.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object.
        """
        return self.transform(dataset)