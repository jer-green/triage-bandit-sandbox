from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .constants import *


class InferenceAlgorithm(Enum):
    RENORMALISE = "renormalise"
    COUNTERFACTUAL = "counterfactual"


class DOSIMode(Enum):
    PRIOR_ADJUSTMENT = "prior_adjustment"
    NOISY_MAX = "noisy_max"


@dataclass
class FeatureFlags:
    """
    max_symptom_prior
        - hyperparm for symptom node leak term calculation
    symptom_leak_proportion
        - hyperparm for symptom node leak term calculation
    single_disease_query
        - adjust the core inference algorithm to only consider terms with a single disease as True
    ignore_other_diseases
        - only perform inference for diseases connected to the evidence
    """

    apply_concept_groups: bool = DEFAULT_APPLY_CONCEPT_GROUPS
    max_symptom_prior: float = DEFAULT_MAX_SYMPTOM_PRIOR
    symptom_leak_proportion: float = DEFAULT_SYMPTOM_LEAK_PROPORTION
    single_disease_query: bool = DEFAULT_SINGLE_DISEASE_QUERY
    ignore_other_diseases: bool = DEFAULT_IGNORE_OTHER_DISEASES
    dosi_mode: DOSIMode = DOSIMode(DEFAULT_DOSI_MODE)
    inference_algorithm: InferenceAlgorithm = InferenceAlgorithm(
        DEFAULT_INFERENCE_ALGORITHM
    )
