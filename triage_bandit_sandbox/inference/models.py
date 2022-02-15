from abc import ABC
from collections import defaultdict
from typing import Dict, List, Mapping, NamedTuple, Optional, Set

import attr
import pandas as pd
from babylon_pgm.constants import LabelType
from babylon_pgm.models.nodes import Node
from babylon_pgm.models.diagnosis import Differential
from dataenforce import Dataset

Priors = Mapping[str, float]
RelativeRisks = Mapping[str, Mapping[str, float]]  # {Disease: {Risk: value}}
SymptomDiseaseMarginals = Mapping[
    str, Mapping[str, float]
]  # {Symptom: {Disease: values}}
SpecialSymptomLeaks = Mapping[str, float]
EvidenceMap = Mapping[str, str]  # concept_id: state


class ConceptGroup(NamedTuple):
    id: str
    name: str
    exclusive: bool
    exhaustive: bool
    concept_ids: List[str]


class ConceptHierarchy(NamedTuple):
    concept_id: str
    groups: List[ConceptGroup]


class Differentials:
    def __init__(self, list_of_differentials: List[Differential]):
        self.df_of_differentials = None
        self.dict_of_differentials = {p.node.id: p for p in list_of_differentials}

    def to_df(self) -> Dataset:
        if self.df_of_differentials is not None:
            return self.df_of_differentials

        node_list = [attr.asdict(p.node) for p in self.dict_of_differentials.values()]
        node_table = pd.DataFrame.from_records(
            node_list, columns=[f.name for f in attr.fields(Node)]
        )
        probability_field_name = attr.fields(Differential)[1].name

        node_table[probability_field_name] = [
            p.probability for p in self.dict_of_differentials.values()
        ]

        self.df_of_differentials = node_table.sort_values(
            by=probability_field_name, ascending=False
        )
        return self.df_of_differentials

    def to_dict(self) -> Dict[str, float]:
        return self.dict_of_differentials

    def __repr__(self):
        # Try to print an instance of this class to see the magic of this method.
        differentials = ", ".join(f"{c!s}" for c in self.dict_of_differentials.values())
        return f"{self.__class__.__name__}({differentials})"

    def __getitem__(self, key: str):
        # This either returns a single item, or a list of items (depending on match)
        try:
            return self.dict_of_differentials[key]
        except KeyError:
            return self.fuzzy_search(key, self.dict_of_differentials)

    @staticmethod
    def fuzzy_search(key: str, dict_to_search: Dict[str, Differential]) -> Differential:
        # TODO: This incurs a O(N) cost per search, we can optimise if that is an issue.
        # TODO: Should we move this to a utils module?
        relevant_keys = [rel_key for rel_key in dict_to_search.keys() if key in rel_key]
        if len(relevant_keys) == 1:
            return dict_to_search[relevant_keys[0]]
        elif len(relevant_keys) > 1:
            return [dict_to_search[rel_key] for rel_key in relevant_keys]
        else:
            return None


@attr.s(auto_attribs=True)
class InferenceModel(ABC):
    nodes: List[Node]
    concept_hierarchies: List[ConceptHierarchy]
    priors: Priors
    symptom_disease_marginals: SymptomDiseaseMarginals
    relative_risks: RelativeRisks
    special_symptom_leaks: SpecialSymptomLeaks

    def __attrs_post_init__(self):
        self._build_mappings()

    def _build_mappings(self):
        disease_to_risk_factors = {
            disease_id: set(relative_risks.keys())
            for disease_id, relative_risks in self.relative_risks.items()
        }
        symptom_to_diseases = {
            symptom_id: set(marginals.keys())
            for symptom_id, marginals in self.symptom_disease_marginals.items()
        }

        # Invert mappings
        risk_factor_to_diseases = defaultdict(set)
        for disease_id, risk_factor_ids in disease_to_risk_factors.items():
            for risk_factor_id in risk_factor_ids:
                risk_factor_to_diseases[risk_factor_id].add(disease_id)
        disease_to_symptoms = defaultdict(set)
        for symptom_id, disease_ids in symptom_to_diseases.items():
            for disease_id in disease_ids:
                disease_to_symptoms[disease_id].add(symptom_id)

        # Concept hierarchies
        self._group_id_to_parent_concept_id = {}
        self._concept_id_to_group_id = {}
        self._group_id_to_group = {}
        for concept_hierarchy in self.concept_hierarchies:
            for group in concept_hierarchy.groups:
                self._group_id_to_parent_concept_id[
                    group.id
                ] = concept_hierarchy.concept_id
                self._group_id_to_group[group.id] = group
                for concept_id in group.concept_ids:
                    self._concept_id_to_group_id[concept_id] = group.id

        self._parent_concept_id_to_concept_hierarchy = {
            concept_hierarchy.concept_id: concept_hierarchy
            for concept_hierarchy in self.concept_hierarchies
        }

        self._id_to_node = {node.id: node for node in self.nodes}
        self._id_to_parents = {**disease_to_risk_factors, **symptom_to_diseases}

        self._id_to_children = {**risk_factor_to_diseases, **disease_to_symptoms}

    def get_parents(self, node_id: str) -> Set[Node]:
        return {
            self.get_node(parent_id)
            for parent_id in self._id_to_parents.get(node_id, {})
            if self.get_node(parent_id)
        }

    def get_children(self, node_id: str) -> Set[Node]:
        return {
            self.get_node(child_id)
            for child_id in self._id_to_children.get(node_id, {})
            if self.get_node(child_id)
        }

    def get_nodes_by_label(self, label: LabelType) -> List[Node]:
        return [node for node in self.nodes if node.label == label]

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._id_to_node.get(node_id)

    def get_prior(self, node_id) -> float:
        return self.priors[node_id]

    def get_symptom_disease_marginal(
        self,
        *,
        symptom_id: str,
        disease_id: str,
    ) -> Optional[float]:
        return self.symptom_disease_marginals.get(symptom_id, {}).get(disease_id)

    def get_symptom_disease_marginals(self, *, symptom_id: str) -> Mapping[str, float]:
        return self.symptom_disease_marginals.get(symptom_id, {})

    def get_relative_risk(
        self, *, risk_factor_id: str, disease_id: str
    ) -> Optional[float]:
        return self.relative_risks.get(disease_id, {}).get(risk_factor_id)

    def get_concept_hierarchy(self, concept_id: str) -> Optional[ConceptHierarchy]:
        return self._parent_concept_id_to_concept_hierarchy.get(concept_id)

    def get_group_parent_concept_id(self, group_id: str) -> str:
        return self._group_id_to_parent_concept_id[group_id]

    def get_group_id(self, concept_id: str) -> Optional[str]:
        return self._concept_id_to_group_id.get(concept_id)

    def get_group(self, group_id: str) -> ConceptGroup:
        return self._group_id_to_group[group_id]

    def get_special_symptom_leak(self, node_id) -> Optional[float]:
        return self.special_symptom_leaks.get(node_id)
