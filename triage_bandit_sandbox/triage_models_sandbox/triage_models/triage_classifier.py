from typing import Optional, List, Union, Tuple
import numpy as np
from sklearn.base import BaseEstimator

from babylon_pgm.triage_models.triage_decisions import TRIAGE_MODEL_DECISION_TYPE, TriageModelDecisionDefault
from babylon_pgm.metrics.triage_metrics import TriageMetrics

from ..triage_data_transformers import (
    BaseDataExtractor,
    MultiLabelEncoder,
)
from ..interfaces import (
    TriageModel,
    TriageDataset,
    TriageDataTransformer,
    PreparedDataTransformer
)


DEFAULT_PREPROCESSING = [
    ('data_extraction', BaseDataExtractor()),
    ('label_encoder', MultiLabelEncoder())
]

class TriageClassifier(TriageModel):
    def __init__(
            self,
            model: BaseEstimator,
            triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
            preprocessing_pipeline: Optional[
                List[Tuple[str, Union[TriageDataTransformer, PreparedDataTransformer]]]
            ] = DEFAULT_PREPROCESSING,
    ):
        """
        :param model: Classifier from scikit-learn.
        :param triage_decisions: Triage decisions to be used (according to world region).
        :param preprocessing_pipeline: List of transformations to be applied to the data
                                        before passing them to the model.
                                        Example: [("transform0", TriageDataTransformer)]
        """
        self._triage_decisions = triage_decisions
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self._params_to_track = {
            "model": model.__class__.__name__,
            "preprocessing": [
                step[1].__class__.__name__
                for step in self.preprocessing_pipeline
            ],
        }

    def __str__(self):
        return "Classifier"

    def fit(
            self,
            data: TriageDataset,
            **kwargs
    ) -> None:
        """
        Train the model.

        :param data: Training data.
        """
        for k, v in kwargs.items():
            self._params_to_track[k] = v
            
        for proc in self.preprocessing_pipeline:
            data = proc[1].fit_transform(data, train=True)
        self.model.fit(data.features, data.correct_decisions, **kwargs)

    def predict(self, data: TriageDataset) -> List[TRIAGE_MODEL_DECISION_TYPE]:
        """
        Predict triage decisions given a TriageDataset object.
        """
        for proc in self.preprocessing_pipeline:
            data = proc[1].transform(data, train=False)

        pred_probs = self.model.predict_proba(data.features)
        if isinstance(pred_probs, list):
            pred_probs_array = np.zeros((len(data), len(pred_probs)))
            for i, probs in enumerate(pred_probs):
                pred_probs_array[:, i] = probs[:, 1]
        elif isinstance(pred_probs, np.ndarray):
            pred_probs_array = pred_probs
        else:
            raise TypeError("Format not supported.")

        decisions = [
            self._triage_decisions.get_by_index(np.argmax(pred_probs_array[i]))
            for i in range(len(data))
        ]
        return decisions


