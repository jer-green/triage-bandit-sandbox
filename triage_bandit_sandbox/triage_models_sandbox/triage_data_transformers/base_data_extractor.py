from typing import List, Optional
import numpy as np

from babylon_pgm.constants import LabelType
from babylon_pgm.triage_models.triage_decisions import (
    TRIAGE_MODEL_DECISION_TYPE,
    TriageModelDecisionDefault,
)
from ..interfaces import (
    TriageDataTransformer,
    TriageDataset,
    PreparedData,
)


class BaseDataExtractor(TriageDataTransformer):
    def __init__(
            self,
            triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
            node_types: Optional[List[LabelType]] = [LabelType.disease],
            include_age: Optional[bool] = True,
    ):
        """
        Convert a TriageDataset object into a PreparedData object.

        :param triage_decisions: Triage decisions to be used (according to world region).
        :param node_types: List of node types whose differential
                            will be included in the triage model features.
        :param include_age: Whether to include age or not in the triage model features.
        """
        self._triage_decisions = triage_decisions
        self.node_types = node_types
        self.include_age = include_age
        self.features_id = []
        self.num_features = 0

    def _extract_correct_decisions(
            self,
            dataset: TriageDataset
    ) -> (List[TRIAGE_MODEL_DECISION_TYPE], TriageDataset):
        correct_decisions = []
        cards_to_remove = []
        for i, case_card in enumerate(dataset):
            card_decisions = []
            if case_card.doctor_outcomes:
                card_decisions = [
                    self._triage_decisions[outcome.triage]
                    for outcome in case_card.doctor_outcomes if outcome.triage
                ]
            elif case_card.judgements:
                card_decisions = [
                    self._triage_decisions[outcome.ideal_triage]
                    for outcome in case_card.judgements
                ]
            if len(card_decisions) > 0:
                correct_decisions.append(card_decisions)
            else:
                cards_to_remove.append(i)
        for i in sorted(cards_to_remove, reverse=True):
            dataset.remove(i)
        return correct_decisions, dataset

    def fit(self, dataset: TriageDataset) -> None:
        """
        Fit the parameters used to transform the data.
        """
        set_of_nodes_id = set()
        for card in dataset:
            card_nodes_id = [
                d.node.id for d in card.differentials
                if d.node.label in self.node_types
            ]
            set_of_nodes_id = set_of_nodes_id.union(set(card_nodes_id))

        self.features_id = list(set_of_nodes_id)
        self.num_features = len(self.features_id)


    def transform(self, dataset: TriageDataset, train: bool) -> PreparedData:
        """
        Transform the data.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object containing only features
                    (differentials of nodes included in self.node_types)
                    and correct_decisions.
        """
        if train:
            correct_decisions, dataset = self._extract_correct_decisions(dataset)
        else:
            correct_decisions = None
        ages_array = np.zeros((len(dataset), 1))
        X = np.zeros((len(dataset), self.num_features))
        for i, card in enumerate(dataset):
            ages_array[i] = card.age / 100
            card_dict = {
                d.node.id: d.probability for d in card.differentials
                if d.probability != None
            }
            X[i] = [card_dict.get(node_id, 0) for node_id in self.features_id]
        if self.include_age:
            X = np.concatenate((X, ages_array), axis=1)
        return PreparedData(
            features=X,
            correct_decisions=correct_decisions,
        )


    def fit_transform(self, dataset: TriageDataset, train: bool) -> PreparedData:
        """
        Fit the parameters for the data transformation and transform the data.
        (Use this method as default to transform data).

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object containing only features
                    (differentials of nodes included in self.node_types)
                    and correct_decisions.
        """
        self.fit(dataset)
        return self.transform(dataset, train)

