from typing import Tuple
from babylon_pgm.constants import LabelType, AGE_MIN, AGE_MAX

CORTEX_LABEL_MAPPING = {
    "DISEASE": LabelType.disease,
    "RISK_FACTOR": LabelType.risk,
    "SYMPTOM": LabelType.symptom,
    "SUPER_NODE": LabelType.super_,
}


AGE_RANGES = [
    (15, 24),
    (25, 39),
    (40, 59),
    (60, 74),
    (75, 100),
]


def _get_age_range(age: int) -> Tuple:
    for r in AGE_RANGES:
        if r[0] <= age <= r[1]:
            return r
    if age < AGE_MIN:
        return AGE_RANGES[0]
    if age > AGE_MAX:
        return AGE_RANGES[-1]