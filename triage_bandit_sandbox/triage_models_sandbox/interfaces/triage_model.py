from abc import ABC
from abc import abstractmethod

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
from scipy.stats import mode
from pathlib import Path
import pickle

from babylon_pgm.triage_models.triage_decisions import TRIAGE_MODEL_DECISION_TYPE, TriageModelDecisionDefault
from babylon_pgm.metrics.triage_metrics import TriageMetrics, MetricsManager

from .triage_data_transformers import TriageDataTransformer, PreparedDataTransformer
from .triage_data import TriageDataset


DEFAULT_PREPROCESSING = [('transform', TriageDataTransformer())]

class TriageModel(ABC):
    """
    Base triage model class.
    """
    def __init__(
            self,
            triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
            preprocessing_pipeline: Optional[
                List[Tuple[str, Union[TriageDataTransformer, PreparedDataTransformer]]]
            ] = DEFAULT_PREPROCESSING
    ):
        """
        :param triage_decisions: Triage decisions to be used (according to world region).
        :param preprocessing_pipeline: List of transformations to be applied to the data
                                        before passing them to the model.
                                        Example: [("transform0", TriageDataTransformer)]
        """
        self._triage_decisions = triage_decisions
        self.preprocessing_pipeline = preprocessing_pipeline
        self._params_to_track = {}

    def __str__(self):
        return "Triage model"

    @property
    def params_to_track(self) -> Dict[str, Any]:
        """
        Return a dictionary of the model parameters to be tracked with MLflow.
        """
        return self._params_to_track

    @abstractmethod
    def fit(self, data: TriageDataset, **kwargs) -> None:
        """
        Train the triage model.
        """
        pass

    @abstractmethod
    def predict(self, data: TriageDataset) -> List[TRIAGE_MODEL_DECISION_TYPE]:
        """
        Predict triage decisions given a TriageDataset object.
        """
        pass

    def score(
            self,
            data: TriageDataset,
            use: str = "judgements",
            how: str = "by_doctor",
            verbose: bool = True,
    ) -> TriageMetrics:
        """
        Get triage metrics for cards in data.

        :param data: TriageDataset
        :param use: str
            Whether to use judgements or outcomes as ground truth.
            Choices= ["judgements", "outcomes"]
        :param how: str
            Whether to compute metrics w.r.t. each doctor or each card.
            Choices= ["by_doctor", "by_card"]
        :return: TriageMetrics object
        """
        assert use in ["judgements", "outcomes"], "use must be either 'judgements' or 'outcomes'."
        assert how in ["by_doctor", "by_card"], "how must be either 'by_doctor' or 'by_card'."
        y_pred = self.predict(data)
        y_pred_ext = []
        y_min = []
        y_max = []
        y_ideal = []
        failed_idxs = []
        for i, case_card_data in enumerate(data.data):
            try:

                if use == "outcomes":
                    correct_decisions = []
                    if case_card_data.doctor_outcomes:
                        correct_decisions = np.array([
                            self._triage_decisions[outcome.triage]
                            for outcome in case_card_data.doctor_outcomes if outcome.triage
                        ])

                    if len(correct_decisions) > 0:
                        if how == "by_card":
                            y_min.append(min(correct_decisions))
                            y_max.append(max(correct_decisions))
                            y_ideal.append(
                                self._triage_decisions.get_by_index(
                                    mode([c.index for c in correct_decisions]).mode[0]
                                )
                            )
                            y_pred_ext.append(y_pred[i])
                        elif how == "by_doctor":
                            y_min.extend(correct_decisions)
                            y_max.extend(correct_decisions)
                            y_ideal.extend(correct_decisions)
                            y_pred_ext.extend([y_pred[i]] * len(correct_decisions))

                elif use == "judgements":
                    if case_card_data.judgements:
                        min_judgements = [
                            self._triage_decisions[judg.minimum_triage]
                            for judg in case_card_data.judgements
                        ]
                        max_judgements = [
                            self._triage_decisions[judg.maximum_triage]
                            for judg in case_card_data.judgements
                        ]
                        ideal_judgements = [
                            self._triage_decisions[judg.ideal_triage]
                            for judg in case_card_data.judgements
                        ]

                        if how == "by_card":
                            y_min.append(min(min_judgements))
                            y_max.append(max(max_judgements))
                            y_ideal.append(
                                self._triage_decisions.get_by_index(
                                    mode([c.index for c in ideal_judgements]).mode[0]
                                )
                            )
                            y_pred_ext.append(y_pred[i])
                        elif how == "by_doctor":
                            y_min.extend(min_judgements)
                            y_max.extend(max_judgements)
                            y_ideal.extend(ideal_judgements)
                            y_pred_ext.extend([y_pred[i]] * len(min_judgements))
            except:
                failed_idxs.append(i)

        metrics = MetricsManager(
            y_min=y_min, y_ideal=y_ideal, y_max=y_max
        ).calculate_metrics(y_pred=y_pred_ext)
        if verbose:
            self._print_metrics(metrics)
        print(f"Triage metrics not computed for cards: {failed_idxs}")
        return metrics


    def save(self, filepath: str, filename: str):
        Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(f"{Path(filepath, filename)}.p", 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str, filename: str):
        with open(f"{Path(filepath, filename)}.p", 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _print_metrics(metrics: TriageMetrics):
        print(f"Accuracy: {metrics.accuracy.score}")
        print(f"Appropriateness: {metrics.appropriateness.score}")
        print(f"Safety: {metrics.safety.score}")
        print(f"Undertriage: {metrics.undertriage.score}")
        print(f"Overtriage: {metrics.overtriage.score}")