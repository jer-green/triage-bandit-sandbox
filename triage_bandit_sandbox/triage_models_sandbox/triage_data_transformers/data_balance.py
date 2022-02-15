from typing import Optional
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.base import BaseSampler

from ..interfaces import PreparedDataTransformer, PreparedData


class OverSampler(PreparedDataTransformer):
    """
    Fix class imbalance by over-sampling data using a
    user-specified sampler method from imblearn package.

    """
    def __init__(self, sampler: BaseSampler):
        self.sampler = sampler

    def fit(self, dataset: PreparedData):
        """
        Fit the parameters used to transform the data.

        """
        X = dataset.features
        y = dataset.correct_decisions
        if len(y.shape) == 1:
            self.sampler.fit(X, y)

    def transform(
            self, dataset: PreparedData, train: Optional[bool] = True
    ) -> PreparedData:
        """
        Transform a PreparedData object into a PreparedData object
        by over-sampling all classes (triage decisions) except the majority one
        in the data.

        :param dataset: Data to be over-sampled.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object
        """
        if train:
            X = dataset.features
            y = dataset.correct_decisions
            if len(y.shape) == 1:
                X, y = self.sampler.fit_resample(X, y)
                dataset.features = X
                dataset.correct_decisions = y
        return dataset

    def fit_transform(
            self, dataset: PreparedData, train: Optional[bool] = True
    ) -> PreparedData:
        """
        Transform a PreparedData object into a PreparedData object
        by over-sampling all classes (triage decisions) except the majority one
        in the data.

        :param dataset: Data to be over-sampled.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object
        """
        return self.transform(dataset)


class SMOTEoversampler(OverSampler):
    """
    Fix class imbalance by over-sampling data
    using SMOTE over-sampler.
    """
    def __init__(self):
        super().__init__(sampler=SMOTE(random_state=42))


class ADASYNoversampler(OverSampler):
    """
    Fix class imbalance by over-sampling data
    using ADASYN over-sampler.
    """
    def __init__(self):
        super().__init__(sampler=ADASYN(
            random_state=42,
            n_neighbors=5
        ))


