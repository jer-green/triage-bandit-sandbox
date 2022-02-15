from typing import List, Dict, Optional

import torch

from babylon_pgm.triage_models.triage_decisions import (
    TRIAGE_MODEL_DECISION_TYPE,
    TriageModelDecisionDefault,
)
from babylon_pgm.models.nodes import Node
from babylon_pgm.models.diagnosis import Differential
from babylon_pgm.constants import DOSI, DOSI_NODE_ID, LabelType
from babylon_pgm.exceptions import ModelError

from ..interfaces import (
    TriageDataset,
    PreparedData,
    TriageDataTransformer,
)


DOSI_TO_LEAK_TRIAGE = {
    DOSI.MINUTES: TriageModelDecisionDefault.SELF_CARE,
    DOSI.HOURS: TriageModelDecisionDefault.SELF_CARE,
    DOSI.DAYS: TriageModelDecisionDefault.GP,
    DOSI.WEEKS: TriageModelDecisionDefault.GP,
    DOSI.MONTHS: TriageModelDecisionDefault.GP,
}
UNKNOWN_DOSI_LEAK_TRIAGE = TriageModelDecisionDefault.GP




class UtilityModelTransformer(TriageDataTransformer):
    """
    Transform a TriageDataset object into a PreparedData object,
    which are then passed to the utility model.
    """
    def __init__(
            self,
            triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
            max_diseases: Optional[int] = 15,
            dosi_to_leak_triage: Optional[Dict] = DOSI_TO_LEAK_TRIAGE,
            unknown_dosi_leak_triage: Optional[TriageModelDecisionDefault] = UNKNOWN_DOSI_LEAK_TRIAGE,

    ):
        """

        :param triage_decisions: Triage decisions to be used (according to world region).
        :param max_diseases: Number of diseases differentials to use as features.
        :param dosi_to_leak_triage: Triage decision for DOSI.
        :param unknown_dosi_leak_triage:
        """
        self._triage_decisions = triage_decisions
        self.max_diseases = max_diseases
        self.dosi_to_leak_triage = dosi_to_leak_triage
        self.unknown_dosi_leak_triage = unknown_dosi_leak_triage


    def _get_node_decision(self, node: Node) -> TRIAGE_MODEL_DECISION_TYPE:
        if not node.triage:
            raise ModelError(f"Missing triage for node {node.id}")
        return self._triage_decisions[node.triage]


    def _encode_decisions(self, decisions_list):
        # Encode a list of decisions into a long tensor
        return torch.LongTensor([decision.index for decision in decisions_list])


    def _extract_disease_differentials(
            self,
            differentials_per_case: List[Differential]
    ) -> List[Differential]:
        return [
            diff for diff in differentials_per_case
            if diff.node.label == LabelType.disease
        ]


    def fit(self, dataset: TriageDataset):
        pass


    def transform(self, dataset: TriageDataset, train: bool) -> PreparedData:
        """
        Transform the data.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object, suited to be fed to the utility model.
        """
        lists_of_differentials = []
        lists_of_evidence_sets = []
        correct_decisions = []
        list_of_ages = []
        for case_card_data in dataset:
            disease_differentials = self._extract_disease_differentials(case_card_data.differentials)
            if train:
                if case_card_data.doctor_outcomes:
                    for outcome in case_card_data.doctor_outcomes:
                        if outcome.triage:
                            lists_of_differentials.append(disease_differentials)
                            lists_of_evidence_sets.append(case_card_data.evidence)
                            correct_decisions.append(
                                self._triage_decisions[outcome.triage]
                            )
                            list_of_ages.append(case_card_data.age)
                elif case_card_data.judgements:
                    for judg in case_card_data.judgements:
                        lists_of_differentials.append(disease_differentials)
                        lists_of_evidence_sets.append(case_card_data.evidence)
                        correct_decisions.append(
                            self._triage_decisions[judg.ideal_triage]
                        )
                        list_of_ages.append(case_card_data.age)
            else:
                lists_of_differentials.append(disease_differentials)
                lists_of_evidence_sets.append(case_card_data.evidence)
                list_of_ages.append(case_card_data.age)

        # Preallocate tensors
        num_differentials = len(lists_of_differentials)
        num_decisions = len(self._triage_decisions)

        likelihoods = torch.Tensor(num_differentials, self.max_diseases).zero_()
        disease_triages = torch.LongTensor(num_differentials, self.max_diseases).zero_()

        for i, differential in enumerate(lists_of_differentials):
            # Reorder by probability
            sorted_differential = sorted(
                differential, key=lambda disease: disease.probability, reverse=True
            )
            for j, disease in enumerate(sorted_differential[:self.max_diseases]):
                disease_decision = self._get_node_decision(disease.node)
                likelihoods[i, j] = disease.probability
                disease_triages[i, j] = disease_decision.index

        # Get leaks
        leak_triages = []
        for evidence_set in lists_of_evidence_sets:
            if evidence_set is None:
                leak_triages.append(self.unknown_dosi_leak_triage)
            else:
                dosi_state = next(
                    (e.state for e in evidence_set if e.node.id == DOSI_NODE_ID), None
                )
                if dosi_state:
                    dosi_value = DOSI(dosi_state)
                    leak_triage = self.dosi_to_leak_triage[dosi_value]
                else:
                    leak_triage = self.unknown_dosi_leak_triage
                leak_triages.append(leak_triage)

        # Add leak term
        leak_probabilities = 1 - likelihoods.sum(1)
        likelihoods = torch.cat((likelihoods, leak_probabilities.unsqueeze(1)), 1)
        leak_triage_indices = [leak_triage.index for leak_triage in leak_triages]
        disease_triages = torch.cat(
            (disease_triages, torch.LongTensor(leak_triage_indices).unsqueeze(1)), 1
        )

        decision_mask = torch.Tensor(
            num_differentials, self.max_diseases + 1, num_decisions,
        ).zero_()
        for k, decision in enumerate(self._triage_decisions):
            decision_mask[:, :, k] = (decision.index < disease_triages).float()

        if train:
            correct_decisions = (
                self._encode_decisions(correct_decisions) if correct_decisions else None
            )

        return PreparedData(
            likelihoods=likelihoods,
            disease_triages=disease_triages,
            decision_mask=decision_mask,
            correct_decisions=correct_decisions,
        )


    def fit_transform(self, dataset: TriageDataset, train: bool = True) -> PreparedData:
        """
        Transform the data.

        :param dataset: Data to be transformed.
        :param train: Whether the data are used to train the model.
        :return: PreparedData object, suited to be fed to the utility model.
        """
        return self.transform(dataset, train=train)



