from abc import ABC
from typing import Optional

from .triage_data import PreparedData, TriageDataset

class TriageDataTransformer(ABC):
    """
    Transform a TriageDataset object into a PreparedData object.
    """
    def __init__(self):
        pass

    def fit(self, dataset: TriageDataset):
        pass

    def transform(self, dataset: TriageDataset, train: bool) -> PreparedData:
        pass

    def fit_transform(self, dataset: TriageDataset, train: bool) -> PreparedData:
        pass




class PreparedDataTransformer(ABC):
    """
    Transform a PreparedData object into a PreparedData object.
    """
    def __init__(self):
        pass

    def fit(self, dataset: PreparedData):
        pass

    def transform(self, dataset: PreparedData, train: Optional[bool] = True) -> PreparedData:
        pass

    def fit_transform(self, dataset: PreparedData, train: Optional[bool] = True) -> PreparedData:
        pass

