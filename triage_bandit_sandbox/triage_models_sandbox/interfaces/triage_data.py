from __future__ import annotations
from typing import Optional, List, Any

from dataclasses import dataclass
import pandas as pd

from babylon_pgm.models.evidence import EvidenceSet
from babylon_pgm.models.diagnosis import Differential

from clinical_validation_sandbox.models import (
    DoctorOutcome,
    Judgement,
    CaseCard,
)

@dataclass
class TriageData:
    """
    Collect all the info from a case card which
    are used to train and test triage models.
    """
    group_id: int
    age: int
    sex: str
    doctor_outcomes: Optional[List[DoctorOutcome]] = None
    judgements: Optional[List[Judgement]] = None
    evidence: Optional[EvidenceSet] = None
    differentials: Optional[List[Differential]] = None

    def to_df(self) -> pd.Series:
        """
        Convert a data sample from a case card
        into a pandas Series.

        :return: pandas Series
        """
        return pd.Series(self.__dict__)

    @classmethod
    def from_case_card(
            cls,
            card: CaseCard,
            differentials: Optional[List[Differential]] = None
    ) -> TriageData:
        """
        Create a TriageData object from a case card.
        """
        if len(card.doctor_outcomes) > 0:
            doctor_outcomes = [outcome for outcome in card.doctor_outcomes]
        else:
            doctor_outcomes = None

        if len(card.judgements) > 0:
            judgements = [judg for judg in card.judgements]
        else:
            judgements = None

        evidence_set = EvidenceSet(
            [item.evidence for item in card.evidence]
        )

        return cls(
            group_id=card.id,
            age=card.age,
            sex=card.sex,
            doctor_outcomes=doctor_outcomes,
            judgements=judgements,
            evidence=evidence_set,
            differentials=differentials,
        )



@dataclass
class TriageDataset:
    """
    Collection of TriageData, that is a dataset to train triage models.
    """
    data: List[TriageData]

    def __len__(self):
        return len(self.data)

    def to_df(self) -> pd.DataFrame:
        """
        Convert it into a pandas DataFrame.
        """
        return pd.DataFrame.from_records(
            [d.__dict__ for d in self.data]
        )

    def __getitem__(self, item: int):
        return self.data[item]

    def remove(self, item: int):
        """
        Remove the data sample at index item.
        """
        del self.data[item]


@dataclass
class PreparedData:
    """
    Data extracted from case cards and processed to be used
    for training and testing of triage models.
    """
    features: Optional[Any] = None
    likelihoods: Optional[Any] = None
    disease_triages: Optional[Any] = None
    decision_mask: Optional[Any] = None
    correct_decisions: Optional[Any] = None

    def __len__(self) -> int:
        if self.features is not None:
            try:
                return self.features.shape[0]
            except:
                try:
                    return len(self.features)
                except:
                    raise TypeError("Format not recognized.")
        elif self.likelihoods is not None:
            return self.likelihoods.shape[0]
        elif self.correct_decisions is not None:
            try:
                return self.correct_decisions.shape[0]
            except:
                try:
                    return len(self.correct_decisions)
                except:
                    raise TypeError("Format not recognized.")
        else:
            return 0
