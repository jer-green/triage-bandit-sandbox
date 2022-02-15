import json
from pathlib import Path
from typing import Any, Dict

from babylon_pgm.constants import LabelType, BOOLEAN_STATES, Gender

from ..inference.models import ConceptHierarchy, ConceptGroup, InferenceModel
from babylon_pgm.models.nodes import Node
from .constants import _get_age_range


def build_inference_model_from_dict(
    inference_model_dict: Dict[str, Any],
    diagnostic_model_dict: Dict[str, Any],
) -> InferenceModel:

    nodes = [
        Node(
            id=node_dict["id"],
            concept=node_dict["concept"],
            grouping_key=None,
            is_searchable=node_dict["searchable"],
            label=LabelType(node_dict["label"]),
            layman_name=node_dict["layman_name"],
            name=node_dict["name"],
            states=BOOLEAN_STATES,
            tags=None,
            cruelty_distribution=node_dict.get("cruelty_distribution"),
            triage=node_dict.get("triage"),
        )
        for node_dict in diagnostic_model_dict["nodes"]
    ]

    relative_risks = inference_model_dict["relative_risks"]

    symptom_disease_marginals = {
        node_id: {parent_id: probability for parent_id, probability in parents.items()}
        for node_id, parents in inference_model_dict[
            "symptom_disease_marginals"
        ].items()
    }

    disease_priors = inference_model_dict["incidence_priors"]
    risk_factor_priors = inference_model_dict["prevalence_priors"]

    # TODO: stop it being optional.
    if "special_symptom_leaks" in inference_model_dict:
        special_symptom_leaks = inference_model_dict["special_symptom_leaks"]
    else:
        special_symptom_leaks = {}

    # Build concept hierarchies
    concept_hierarchies = [
        ConceptHierarchy(
            concept_id=item["concept"]["id"],
            groups=[
                ConceptGroup(
                    name=group["name"],
                    id=group["id"],
                    exclusive=group["exclusive"],
                    exhaustive=group["exhaustive"],
                    concept_ids=[concept["id"] for concept in group["concepts"]],
                )
                for group in item["groups"]
            ],
        )
        for item in inference_model_dict.get("concept_groups", [])
    ]

    model = InferenceModel(
        nodes=nodes,
        concept_hierarchies=concept_hierarchies,
        priors={**disease_priors, **risk_factor_priors},
        symptom_disease_marginals=symptom_disease_marginals,
        relative_risks=relative_risks,
        special_symptom_leaks=special_symptom_leaks,
    )

    return model


def load_json_object(json_path: Path) -> Dict:
    """Attempts to load a single JSON object from a file.

    Raises a TypeError if the JSON parsing is successful but the file contained
    a JSON array.

    Keyword arguments:
    json_path (Path): Path to file to load a json object from.
    """
    with json_path.open("r") as infile:
        json_dict = json.load(infile)
        if not isinstance(json_dict, dict):
            raise TypeError("Contents of ", json_path, " is not one JSON object")

        return json_dict


class JSONDatasetToInferenceModelConverter:
    def __init__(
        self,
        models_dir: Path,
        model_version: str,
    ):
        self.models_dir = models_dir
        self.model_version = model_version
        self.model_cache = {}

    def get_model(
        self,
        age: int,
        sex: Gender,
    ) -> InferenceModel:

        gender_str = sex.value.upper()
        assert isinstance(age, int), "age should be an integer"

        age_range = _get_age_range(age)
        if (age_range, gender_str) not in self.model_cache:
            self.model_cache[(age_range, gender_str)] = self.generate_model(age, sex)

        return self.model_cache[(age_range, gender_str)]

    def generate_model(self, age: int, sex: Gender) -> InferenceModel:
        min_age, max_age = _get_age_range(age)
        inference_model_file = (
            self.models_dir
            / "components/inference_engine"
            / self.model_version
            / "perov"
            / f"model.{sex.value.lower()}_only.{min_age}.{max_age}.json"
        )
        diagnostic_model_file = (
            self.models_dir
            / "components/diagnostic_engine"
            / self.model_version
            / f"diagnostic_model.{sex.value.lower()}_only.{min_age}.{max_age}.json"
        )

        return build_inference_model_from_dict(
            load_json_object(inference_model_file),
            load_json_object(diagnostic_model_file),
        )